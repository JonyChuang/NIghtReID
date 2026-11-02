"""CPU integration check only: no training loop or optimizer step is called."""
import argparse
import io
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    import torch
    from torch.nn import functional as F
    from config import cfg as base_cfg
    from model.make_model import add_multilayer_config, make_model as original_model
    from solver import make_optimizer
    from solver.scheduler_factory import create_scheduler
    from loss.triplet_loss import TripletLoss
    from .model import add_cnl_config, make_model
    from .global_cnl import GlobalCNL
    from .loss import make_loss
    from .processor import save_checkpoint, evaluate
    from .indexed_data import indexed_loader
    from datasets import make_dataloader

    torch.set_num_threads(2)
    cfg = add_cnl_config(base_cfg.clone())
    cfg.merge_from_file(str(Path(__file__).parent / 'configs/night600.yml'))
    cfg.freeze()
    assert cfg.TEST.NECK_FEAT == 'before' and cfg.TEST.IMS_PER_BATCH == 256
    assert 'HEAD_LR_FACTOR' not in cfg.SOLVER and 'MLDC_HEAD_LR_FACTOR' not in cfg.SOLVER
    assert (cfg.SOLVER.MAX_EPOCHS, cfg.SOLVER.IMS_PER_BATCH, cfg.SOLVER.WARMUP_EPOCHS) == (120, 64, 20)
    torch.manual_seed(cfg.SOLVER.SEED)
    model = make_model(cfg, 300, 8, 1)
    # The project global branch is the reference, with its later fusion disabled.
    reference_cfg = cfg.clone()
    reference_cfg.defrost()
    reference_cfg = add_multilayer_config(reference_cfg)
    reference_cfg.MODEL.MULTILAYER_FUSION = False
    reference_cfg.freeze()
    torch.manual_seed(cfg.SOLVER.SEED)
    reference = original_model(reference_cfg, 300, 8, 1)
    assert model.state_dict().keys() == reference.state_dict().keys()
    assert all(torch.equal(v, reference.state_dict()[k]) for k, v in model.state_dict().items())
    assert not any(any(s in k for s in ('mldc', 'fusion', 'enhancer')) for k in model.state_dict())
    assert model.base.norm.weight.requires_grad and model.base.norm.bias.requires_grad
    generator = torch.Generator().manual_seed(43)
    images = torch.randn(4, 3, *cfg.INPUT.SIZE_TRAIN, generator=generator)
    model.eval();reference.eval()
    captured = {}
    def capture(module, inputs, output):
        captured['last_tokens'] = output.detach()
    hook = model.base.blocks[-1].register_forward_hook(capture)
    with torch.no_grad():
        features = model(images)
        expected = reference(images)
        torch.testing.assert_close(features, expected, rtol=0, atol=0)
        torch.testing.assert_close(features, model.base.norm(captured['last_tokens'])[:, 0], rtol=0, atol=0)
    hook.remove()
    del reference, captured
    print('Original SSL initialization, final LayerNorm, head and before-BN retrieval match exactly.', flush=True)

    model.train()
    labels = torch.tensor([0, 0, 1, 1])
    cameras = torch.tensor([0, 1, 0, 1])
    indices = torch.arange(4)
    scores, raw, unit = model(images, labels, return_cnl_features=True)
    torch.testing.assert_close(unit, F.normalize(raw.float(), dim=1), rtol=0, atol=0)
    loss_fn, center = make_loss(cfg, 300)
    original_loss = loss_fn(scores, raw, labels, cameras)
    original_formula = F.cross_entropy(scores.float(), labels) + TripletLoss()(raw.float(), labels)[0]
    torch.testing.assert_close(original_loss, original_formula, rtol=0, atol=0)
    memory = GlobalCNL(labels.repeat(2), torch.cat((cameras, 1-cameras)),
        gamma=cfg.SOLVER.CNL_GAMMA, temperature=cfg.SOLVER.CNL_TEMPERATURE,
        momentum=cfg.SOLVER.CNL_MOMENTUM)
    memory.update(unit.detach(), indices+4, labels, 1-cameras)
    old_features, old_valid = memory.features.clone(), memory.valid.clone()
    cnl, stats = memory(unit, indices, labels, cameras)
    assert stats['active_anchors'] == 4 and cnl.item() > 0
    assert torch.equal(memory.features, old_features) and torch.equal(memory.valid, old_valid)
    parameters = [model.base.norm.weight, model.base.patch_embed.proj.weight]
    gradients = torch.autograd.grad(cnl, parameters, retain_graph=True)
    norms = [float(g.norm()) for g in gradients]
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients)
    total = original_loss + cfg.SOLVER.CNL_WEIGHT * cnl
    total.backward()
    assert model.classifier.weight.grad is not None and torch.isfinite(model.classifier.weight.grad).all()
    # Explicitly exercise memory first-fill and momentum on synthetic slots only.
    memory.update(unit.detach(), indices, labels, cameras)
    assert memory.valid.all()
    changed = F.normalize(unit.detach()+.01*torch.randn(unit.shape, generator=generator), dim=1)
    wanted = F.normalize(.2*memory.features[indices]+.8*changed, dim=1)
    memory.update(changed, indices, labels, cameras)
    torch.testing.assert_close(memory.features[indices], wanted)
    print('CNL has active cross-camera positives and finite gradients through the original backbone.', flush=True)

    optimizer, _ = make_optimizer(cfg, model, center)
    named = dict(model.named_parameters())
    groups = {id(g['params'][0]): g for g in optimizer.param_groups}
    for name, parameter in named.items():
        if parameter.requires_grad:
            expected_lr = cfg.SOLVER.BASE_LR * (cfg.SOLVER.BIAS_LR_FACTOR if 'bias' in name else 1)
            assert groups[id(parameter)]['lr'] == expected_lr
    scheduler = create_scheduler(cfg, optimizer)
    lr_trace = []
    for epoch in range(121):
        # Initialization gives phase 0; each completed epoch steps to its phase.
        if epoch:
            scheduler.step(epoch)
        lr_trace.append([groups[id(named[k])]['lr'] for k in
                         ('base.cls_token', 'base.blocks.0.norm1.bias', 'classifier.weight', 'bottleneck.weight')])
    assert all(math.isclose(lr, 4e-6) for lr in lr_trace[0])
    assert all(math.isclose(lr, 8e-7) for lr in lr_trace[120])
    for row in lr_trace:
        assert row[0] == row[2] == row[3]
    buffer = io.BytesIO()
    save_checkpoint(model, cfg, 0, buffer, memory=memory, dataset_fingerprint='synthetic_check_only')
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    memory.load_state_dict(checkpoint['cnl_memory'], strict=True)
    assert checkpoint['variant'] == 'transreid_ssl_cnl_20261022'
    assert checkpoint['retrieval'] == 'before'
    del checkpoint, buffer

    # Real dataset wiring and original evaluation routine, without training data iteration.
    loader_cfg = cfg.clone();loader_cfg.defrost();loader_cfg.DATALOADER.NUM_WORKERS = 0;loader_cfg.freeze()
    train_loader, _, val_loader, nq, nc, _, _ = make_dataloader(loader_cfg)
    indexed, dataset = indexed_loader(train_loader)
    assert indexed.batch_sampler is train_loader.batch_sampler
    image, pid, camera, view, index = dataset[0]
    assert tuple(image.shape) == (3, 256, 128) and index == 0
    assert (pid, camera, view) == tuple(dataset.entries[0][1:4])
    # Small evaluation with valid cross-camera matches exercises distance/AP handling.
    # Ten gallery images make Rank-10 defined under the original evaluator.
    gallery_labels = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    batch = (torch.cat((images[:2], images.repeat(3, 1, 1, 1)[:10])),
             [0, 1]+gallery_labels, [0, 0]+[1]*10,
             torch.tensor([0, 0]+[1]*10), torch.zeros(12, dtype=torch.long), ['synthetic']*12)
    metrics = evaluate(cfg, model, [batch], 2, torch.device('cpu'))
    assert all(0 <= value <= 1 for value in metrics.values())
    result = dict(all_passed=True, device='cpu', optimizer_steps=0, training_started=False,
        original_model_state_identical=True, original_global_output_max_abs_difference=0,
        original_final_layernorm_active=True, original_identity_triplet_identical=True,
        original_optimizer_groups=True, original_scheduler_phases_checked=121,
        cnl_feature='L2-normalized before-BN original global descriptor',
        cnl_loss=float(cnl.detach()), cnl_stats=stats,
        cnl_gradient_norms=dict(final_layernorm=norms[0], patch_projection=norms[1]),
        checkpoint_and_memory_roundtrip=True, training_images=len(dataset),
        training_dataset_fingerprint=dataset.fingerprint, classes=nc, queries=nq,
        real_dataset_wiring=True, evaluation_wiring=True,
        lr_phase_0=lr_trace[0], lr_phase_20=lr_trace[20], lr_phase_120=lr_trace[120])
    Path(args.report).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('VERIFICATION_RESULT '+json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
