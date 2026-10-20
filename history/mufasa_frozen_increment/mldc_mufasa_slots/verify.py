"""CPU/GPU integration checks; no epoch loop and no experiment checkpoint."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    import torch.nn.functional as F
    from config import cfg as base_cfg
    from datasets import make_dataloader
    from mldc_mufasa_slots.config import add_config, validate
    from mldc_mufasa_slots.model import (FrozenMLDCMUFASA, MaskAssignment, SetReadout,
                                        compute_loss, residual_feature)
    from mldc_mufasa_slots.train import (assert_frozen, frozen_snapshot, learning_rate,
                                       load_increment, save_checkpoint, prepare_resume, append_row)
    from mldc_mufasa_slots.evaluation import diagnostic_values, query_results, evaluate
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--report', default='')
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(1234)
    cfg = add_config(base_cfg.clone())
    cfg.merge_from_file('mldc_mufasa_slots/variants/slot_residual_joint_bnneck/config.yml')
    validate(cfg)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.TEST.IMS_PER_BATCH = 8
    cfg.freeze()
    device = torch.device(args.device)
    report = {'device': str(device), 'checks': {}}

    # Non-square 128-token masks, known permutation and empty-slot ties.
    matcher = MaskAssignment(4).to(device)
    labels = torch.arange(128, device=device) % 4
    first = F.one_hot(labels, 4).float().unsqueeze(0)
    permutation = torch.tensor([2, 0, 3, 1], device=device)
    second = first[:, :, permutation]
    order, before, after = matcher(first, second)
    assert torch.equal(second.gather(2, order[:, None, :].expand_as(second)), first)
    assert before.item() == 0 and after.item() == 1
    empty = torch.zeros_like(first)
    empty[..., 0] = 1
    order2, _, score2 = matcher(empty, empty)
    assert torch.equal(order2[0], torch.arange(4, device=device)) and torch.isfinite(score2).all()
    report['checks']['exact_mask_assignment_128_tokens'] = True

    readout = SetReadout(256).to(device)
    g, slots = torch.randn(3, 768, device=device), torch.randn(3, 4, 256, device=device)
    r, attention = readout(g, slots)
    reordered, _ = readout(g, slots[:, permutation])
    torch.testing.assert_close(r, reordered, atol=2e-6, rtol=2e-5)
    assert torch.equal(residual_feature(g, r, 0), F.normalize(g, dim=-1))
    assert learning_rate(cfg, 1) == cfg.SOLVER.BASE_LR*cfg.SOLVER.WARMUP_FACTOR
    assert abs(learning_rate(cfg, 10)-cfg.SOLVER.BASE_LR) < 1e-12
    assert 0 < learning_rate(cfg, 120) < learning_rate(cfg, 60)
    report['checks']['permutation_invariant_readout_and_zero_residual'] = True

    # Original evaluation exclusion: same ID and camera is filtered, other IDs remain.
    example = query_results(np.asarray([[0., .1, .2]]), [7, 7, 8, 7], [0, 0, 1, 1], 1,
                            ['query', 'same_camera_positive', 'negative', 'positive'])
    assert len(example) == 1 and example[0]['AP'] == .5 and example[0]['rank1'] == 0
    report['checks']['query_diagnostics_same_camera_exclusion'] = True

    train_loader, _, val_loader, nq, nc, cams, views = make_dataloader(cfg)
    model = FrozenMLDCMUFASA(cfg, nc, cams, views).to(device)
    snapshot = frozen_snapshot(model)
    images, targets, camera, view = next(iter(train_loader))
    images, targets = images.to(device), targets.to(device)
    camera, view = camera.to(device), view.to(device)
    with torch.no_grad():
        original = model.baseline(images, cam_label=camera, view_label=view)
    model.train()
    parts = model(images, cam_label=camera, view_label=view)
    assert torch.equal(original, parts['global_feature'])
    assert parts['feature'].shape == (8, 768)
    assert parts['maps'].shape == (8, 4, 128, 4)
    assert parts['reconstruction'].shape == (8, 128, 768)
    torch.testing.assert_close(parts['feature'].norm(dim=-1), torch.ones(8, device=device))
    assert not parts['target'].requires_grad
    loss, terms = compute_loss(cfg, parts, targets)
    loss.backward()
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    norms = {}
    for name, p in trainable.items():
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise AssertionError('Missing/non-finite new-module gradient: '+name)
        norms[name] = p.grad.norm().item()
    for prefix in ['slots.encoders.0.', 'slots.encoders.1.', 'slots.encoders.2.', 'slots.encoders.3.',
                   'slots.projection.', 'readout.', 'decoder.', 'classifier.']:
        assert sum(value for name, value in norms.items() if name.startswith(prefix)) > 0, prefix
    before_parameter = model.readout.value.weight.detach().clone()
    optimizer = torch.optim.Adam(trainable.values(), lr=.0004)
    optimizer.step()
    assert not torch.equal(before_parameter, model.readout.value.weight)
    assert_frozen(model, snapshot)
    report['checks']['real_PK_batch_backward_all_new_modules'] = True
    report['checks']['original_parameters_buffers_and_eval_mode_unchanged'] = True
    report['losses'] = {name: float(value.detach()) for name, value in terms.items()}
    report['trainable_parameters'] = sum(p.numel() for p in trainable.values())
    del parts, loss, terms
    model.eval()
    with torch.no_grad():
        p1 = model(images, cam_label=camera, view_label=view, return_parts=True)
        p2 = model(images, cam_label=camera, view_label=view, return_parts=True)
        assert torch.equal(p1['feature'], p2['feature'])
        assert 'reconstruction' not in p1 and 'logits' not in p1
        for value in diagnostic_values(p1).values():
            assert value.shape == (8,) and torch.isfinite(value).all()
        model.scale = 0
        zero = model(images, cam_label=camera, view_label=view)
        assert torch.equal(zero, F.normalize(original, dim=-1))
        model.scale = cfg.MUFASA.RESIDUAL_SCALE
    report['checks']['deterministic_retrieval_and_training_only_decoder'] = True
    report['checks']['zero_residual_recovers_original_normalized_descriptor'] = True

    # Exercise actual checkpoint writer/loader, without retaining any trained weights.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='mufasa_verify_') as temporary:
        checkpoint_path = Path(temporary) / 'increment_20.pth'
        save_checkpoint(cfg, model, optimizer, 20, None, checkpoint_path)
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        from yacs.config import CfgNode as CN
        restored_cfg = CN.load_cfg(saved['config'])
        validate(restored_cfg)
        assert restored_cfg.MODEL.DEVICE_ID == '0'
        assert not any(k.startswith('baseline.') for k in saved['state_dict'])
        load_increment(model, saved)
        with torch.no_grad():
            restored = model(images, cam_label=camera, view_label=view)
        torch.testing.assert_close(restored, p1['feature'], rtol=0, atol=0)
        # An interrupted run may contain CSV rows after its latest checkpoint.
        for name in ['training.csv', 'evaluation.csv', 'diagnostics.csv']:
            append_row(Path(temporary) / name, {'epoch': 20, 'value': 1})
            append_row(Path(temporary) / name, {'epoch': 21, 'value': 2})
        prepare_resume(Path(temporary), 20)
        import csv
        with (Path(temporary) / 'training.csv').open() as f:
            assert [int(row['epoch']) for row in csv.DictReader(f)] == [20]
        assert len(list(Path(temporary).glob('resume_archive_*/training.csv'))) == 1
    report['checks']['increment_checkpoint_roundtrip'] = True
    report['checks']['resume_configuration_and_partial_log_archive'] = True
    assert_frozen(model, snapshot)

    # Check both advertised structural controls through the same readout.
    with torch.no_grad():
        g, patches, target = model.extract_frozen(images, camera, view)
        model.slots.match = False
        no_match = model.refine(g, patches)
        model.slots.match = True
        model.slots.fusion = 'mean'
        mean_fusion = model.refine(g, patches)
        model.slots.fusion = 'mfusion'
        single_layer = model.refine(g, [patches[1]]*4)
        for parts in [no_match, mean_fusion, single_layer]:
            assert parts['feature'].shape == (8, 768) and torch.isfinite(parts['feature']).all()
    report['checks']['single_layer_no_matching_and_mean_fusion_controls'] = True
    # Small retrieval fixture covers the complete evaluator, CSVs and visualization.
    from torch.utils.data import DataLoader, Subset
    from datasets.make_dataloader import val_collate_fn
    from mldc_cls_readout.processor.visualization import fixed_pairs
    from mldc_mufasa_slots.visualization import save_visualizations
    records = val_loader.dataset.dataset
    pairs = fixed_pairs(records, nq, 2, 1234)
    ids = {p['pid'] for p in pairs}
    distractors = [i for i in range(nq, len(records)) if records[i][1] not in ids][:10]
    indices = [p['query_index'] for p in pairs]+[p['reference_index'] for p in pairs]+distractors
    fixture = DataLoader(Subset(val_loader.dataset, indices), batch_size=8, collate_fn=val_collate_fn)
    with tempfile.TemporaryDirectory(prefix='mufasa_eval_verify_') as temporary:
        result = evaluate(cfg, model, fixture, 2, Path(temporary) / 'evaluation')
        assert result['diagnostics']['num_images'] == 14
        assert (Path(temporary) / 'evaluation' / 'query_changes.csv').exists()
        vis_cfg = cfg.clone()
        vis_cfg.defrost()
        vis_cfg.OUTPUT_DIR = temporary
        vis_cfg.MUFASA.VIS_SAMPLES = 1
        vis_cfg.freeze()
        save_visualizations(vis_cfg, model, val_loader, nq, 0)
        image_path = Path(temporary) / 'visualizations' / 'epoch_000' / 'q00810.png'
        assert image_path.exists()
        if args.report:
            import shutil
            shutil.copy2(image_path, Path(args.report).with_name('verification_visualization.png'))
    report['checks']['complete_evaluation_CSV_and_fixed_pair_visualization'] = True
    report['num_query'] = nq
    report['num_gallery'] = len(val_loader.dataset)-nq
    report['full_retrieval_evaluation'] = 'Deferred to CUDA training preflight; this script does not train epochs.'
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
