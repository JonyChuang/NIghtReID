"""Controlled CPU checks; no dataset, optimizer steps, or training loop.

python -m mldc_information_preservation.verify --full-model --report verification_report.json
The optional full-model check uses generated images and skips SSL loading.
"""
import argparse
import io
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from .information_preservation import MultiLayerInformationPreservation as Preservation


def vector(cosine):
    return torch.tensor([cosine, math.sqrt(max(0., 1 - cosine ** 2)), 0.])


def fixture(student_positive=.6, student_negative=.7, teacher_positive=.9, teacher_negative=.5):
    memory = Preservation([0, 0, 1], dim=3)
    keys = torch.stack([vector(student_positive), vector(student_negative)])
    layers = torch.stack([vector(teacher_positive).repeat(4, 1), vector(teacher_negative).repeat(4, 1)])
    memory.update(keys, layers, torch.tensor([1, 2]), torch.tensor([0, 1]))
    return memory, vector(1.).reshape(1, 3), vector(1.).repeat(1, 4, 1), torch.tensor([0]), torch.tensor([0])


class PreservationChecks(unittest.TestCase):
    def test_exact_hinge_and_detached_references(self):
        memory, query, layers, indices, labels = fixture()
        query.requires_grad_()
        layers.requires_grad_()
        before = {k: v.clone() for k, v in memory.state_dict().items()}
        loss, stats = memory(query, layers, indices, labels)
        # Teacher margin .4 is capped at .2; student margin is -.1.
        self.assertAlmostEqual(float(loss), .3, places=5)
        self.assertEqual(stats['reliable_triplets'], 1)
        self.assertEqual(stats['violated_triplets'], 1)
        loss.backward()
        self.assertGreater(float(query.grad.abs().sum()), 0)
        self.assertIsNone(layers.grad)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in memory.state_dict().items()))
        self.assertTrue(all(not v.requires_grad for v in memory.buffers()))

    def test_no_penalty_when_fusion_satisfies_reference(self):
        memory, *args = fixture(student_positive=.9, student_negative=.5)
        loss, stats = memory(*args)
        self.assertEqual(float(loss), 0.)
        self.assertEqual(stats['reliable_triplets'], 1)
        self.assertEqual(stats['violated_triplets'], 0)

    def test_unreliable_layers_provide_no_target(self):
        for pos, neg in ((.51, .5), (.3, .5)):
            memory, query, layers, indices, labels = fixture(teacher_positive=pos, teacher_negative=neg)
            query.requires_grad_()
            loss, stats = memory(query, layers, indices, labels)
            self.assertEqual(float(loss), 0.)
            self.assertEqual(stats['reliable_triplets'], 0)
            loss.backward()
            self.assertEqual(float(query.grad.abs().sum()), 0.)

    def test_complementary_layers_keep_both_positives(self):
        memory = Preservation([0, 0, 0, 1], dim=3)
        first = torch.stack([vector(.9), vector(.3), vector(.3), vector(.3)])
        second = torch.stack([vector(.3), vector(.9), vector(.3), vector(.3)])
        memory.update(torch.stack([vector(.6), vector(.6), vector(.7)]),
                      torch.stack([first, second, vector(.5).repeat(4, 1)]),
                      torch.tensor([1, 2, 3]), torch.tensor([0, 0, 1]))
        loss, stats = memory(vector(1.)[None], vector(1.).repeat(1, 4, 1), torch.tensor([0]), torch.tensor([0]))
        self.assertEqual(stats['available_positives'], 2)
        self.assertEqual(stats['reliable_triplets'], 2)
        self.assertEqual(stats['reference_layer_0'], 1)
        self.assertEqual(stats['reference_layer_1'], 1)
        self.assertAlmostEqual(float(loss), .3, places=5)

    def test_hard_negative_union_and_identity_mask(self):
        memory = Preservation([0, 0, 1, 2], dim=3, hard_negatives=1)
        fused = torch.tensor([1., .9, .1, .2])
        layers = torch.tensor([[1., .1, .9, .2], [1., .1, .2, .9]] * 2)
        selected = memory.select_negatives(fused, layers, torch.tensor([False, False, True, True]))
        self.assertEqual(selected.tolist(), [2, 3])

    def test_duplicate_original_image_never_becomes_positive(self):
        memory = Preservation([0, 1], dim=3)
        query = torch.stack([vector(.8), vector(.7), vector(.4)]).requires_grad_()
        layers = query.detach()[:, None].repeat(1, 4, 1)
        loss, stats = memory(query, layers, torch.tensor([0, 0, 1]), torch.tensor([0, 0, 1]))
        self.assertEqual(stats['available_positives'], 0)
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(query.grad).all())

    def test_current_features_replace_stale_entries(self):
        memory, query, layers, indices, labels = fixture()
        memory.update(-query, -layers, indices, labels)
        one, stats = memory(query, layers, indices, labels)
        memory.features[0].fill_(0.)
        memory.layer_features[0].fill_(0.)
        two, stats2 = memory(query, layers, indices, labels)
        self.assertEqual(stats['candidate_images'], 3)
        self.assertEqual(stats2['candidate_images'], 3)
        self.assertTrue(torch.equal(one, two))

    def test_momentum_duplicate_order_and_roundtrip(self):
        memory, query, layers, indices, labels = fixture()
        memory.update(query, layers, indices, labels)
        new_query = vector(.5)[None]
        new_layers = new_query[:, None].repeat(1, 4, 1)
        expected = F.normalize(.2 * query + .8 * new_query, dim=-1)
        memory.update(new_query, new_layers, indices, labels)
        self.assertTrue(torch.allclose(memory.features[0], expected[0]))
        duplicate_features = torch.stack([vector(.1), vector(.6), vector(.3)])
        duplicate_indices = torch.tensor([0, 0, 2])
        duplicate_labels = torch.tensor([0, 0, 1])
        a, b = Preservation([0, 0, 1], dim=3), Preservation([0, 0, 1], dim=3)
        a.update(duplicate_features, duplicate_features[:, None].repeat(1, 4, 1), duplicate_indices, duplicate_labels)
        order = torch.tensor([2, 1, 0])
        b.update(duplicate_features[order], duplicate_features[order, None].repeat(1, 4, 1), duplicate_indices[order], duplicate_labels[order])
        self.assertTrue(torch.allclose(a.features, b.features))
        self.assertTrue(torch.allclose(a.layer_features, b.layer_features))
        buffer = io.BytesIO()
        torch.save(memory.state_dict(), buffer)
        buffer.seek(0)
        restored = Preservation([0, 0, 1], dim=3)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        self.assertTrue(all(torch.equal(v, restored.state_dict()[k]) for k, v in memory.state_dict().items()))

    def test_missing_positive_or_negative_returns_finite_zero(self):
        for ids in ([0, 0], [0, 1]):
            memory = Preservation(ids, dim=3)
            query = torch.stack([vector(.8), vector(.6)]).requires_grad_()
            loss, stats = memory(query, query.detach()[:, None].repeat(1, 4, 1), torch.arange(2), torch.tensor(ids))
            self.assertEqual(float(loss), 0.)
            self.assertEqual(stats['active_anchors'], 0)
            loss.backward()
            self.assertTrue(torch.isfinite(query.grad).all())


def full_model_check():
    from config import cfg as base_cfg
    from .model.make_model_mldc import MLDCReID, add_mldc_config
    from .model.backbones.vit_pytorch_mldc import TransReID
    from .loss.make_loss_mldc import make_loss
    # Import the actual training/checkpoint integration, but do not call do_train.
    from .processor.processor_mldc import save_checkpoint
    torch.manual_seed(1234)
    cfg = add_mldc_config(base_cfg.clone())
    cfg.merge_from_file(str(Path(__file__).parent / 'configs/night600.yml'))
    with patch.object(TransReID, 'load_param', lambda *args, **kwargs: None):
        model = MLDCReID(cfg, num_classes=2, camera_num=2, view_num=1)
    model.train()
    images = torch.randn(4, 3, *cfg.INPUT.SIZE_TRAIN)
    labels, indices = torch.tensor([0, 0, 1, 1]), torch.arange(4)
    scores, raw, unit, layers = model(images, return_preservation_features=True)
    assert scores.shape == (4, 2) and raw.shape == unit.shape == (4, 768)
    assert layers.shape == (4, 4, 768) and not layers.requires_grad
    memory = Preservation([0, 0, 1, 1, 0, 2], dim=768)
    # Controlled memory keys guarantee a reliable teacher and a deficient fused
    # margin. These are fixtures, not observed image features or retrieval results.
    direction = unit[0].detach().roll(1)
    keys = F.normalize(torch.stack([-unit[0].detach() + .3 * direction,
                                    unit[0].detach() - .1 * direction]), dim=-1)
    memory.update(keys, torch.stack([layers[0], -layers[0]]), torch.tensor([4, 5]), torch.tensor([0, 2]))
    state = {k: v.clone() for k, v in memory.state_dict().items()}
    preservation, stats = memory(unit, layers, indices, labels)
    assert preservation > 0 and stats['violated_triplets'] > 0
    gradient = torch.autograd.grad(preservation, model.mldc.layer_weights, retain_graph=True)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 1e-10
    classifier_gradient = torch.autograd.grad(preservation, model.classifier.weight, retain_graph=True, allow_unused=True)[0]
    assert classifier_gradient is None
    loss_fn, _ = make_loss(cfg, num_classes=2)
    base = loss_fn(scores, raw, labels)
    total = base + cfg.SOLVER.PRESERVE_WEIGHT * preservation
    total.backward()
    for prefix in ('base.blocks.', 'base.patch_embed.', 'mldc.', 'bottleneck.', 'classifier.'):
        norms = [float(p.grad.abs().sum()) for n, p in model.named_parameters()
                 if n.startswith(prefix) and p.grad is not None]
        assert norms and sum(norms) > 0, prefix
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.equal(v, state[k]) for k, v in memory.state_dict().items())
    model.eval()
    with torch.no_grad():
        descriptor = model(images[:1])
    assert descriptor.shape == (1, 768)
    # Small stub exercises the real checkpoint writer without serializing the
    # entire ViT. The actual model forward/backward is exercised above.
    class CheckpointStub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mldc = model.mldc
            self.train_feature, self.readout = model.train_feature, model.readout
    buffer = io.BytesIO()
    save_checkpoint(CheckpointStub(), cfg, 0, buffer, memory=memory, dataset_fingerprint='synthetic-check')
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=False)
    assert 'preservation_memory' in saved and 'cnl_memory' not in saved
    return dict(full_vit_forward_backward=True, image_shape=list(images.shape),
                descriptor_dim=768, preservation_only_fusion_gradient=gradient.tolist(),
                preservation=float(preservation.detach()), total=float(total.detach()),
                memory_unchanged_by_forward_and_backward=True, checkpoint_schema_checked=True,
                validation_scope='Generated CPU images, SSL loading skipped, controlled memory keys; no retrieval evaluation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-model', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PreservationChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    report = dict(unit_checks=result.testsRun, all_passed=True, training_loop_started=False,
                  optimizer_steps=0, cuda_available=torch.cuda.is_available())
    if args.full_model:
        report.update(full_model_check())
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
