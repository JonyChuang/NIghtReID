"""Joint integration validation: forward/backward only, never optimizer.step."""
import argparse
import csv
import gc
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from config import cfg as base_cfg
    from datasets import make_dataloader
    from solver import make_optimizer
    from solver.scheduler_factory import create_scheduler
    from mldc_cls_readout.model.make_model_mldc import MLDCReID, add_mldc_config, configure_head_learning_rates
    from mldc_cls_readout.loss.make_loss_mldc import make_loss
    from mldc_mufasa_slots.config import add_config, validate
    from mldc_mufasa_slots.model import JointMLDCMUFASA, compute_loss, residual_metric_loss, residual_feature
    from mldc_mufasa_slots.train import make_training, CHECKPOINT_KIND, load_joint, save_checkpoint
    from mldc_mufasa_slots.evaluation import diagnostic_values, query_results, evaluate
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--report', default='')
    args = parser.parse_args()
    torch.set_num_threads(8)
    cfg = add_config(base_cfg.clone())
    cfg.merge_from_file(str(Path(__file__).resolve().parent / 'configs/night600.yml'))
    validate(cfg)
    reference = add_mldc_config(base_cfg.clone())
    reference.merge_from_file('mldc_cls_readout/variants/all_cls_random_initialization/config.yml')
    for section in ['INPUT', 'DATASETS', 'DATALOADER', 'SOLVER']:
        assert getattr(cfg, section).dump() == getattr(reference, section).dump(), section
    for key in ['PRETRAIN_PATH', 'PRETRAIN_CHOICE', 'PRETRAIN_HW_RATIO', 'STEM_CONV',
                'STRIDE_SIZE', 'MLDC_TEMPERATURE', 'DROP_PATH', 'DROP_OUT', 'ATT_DROP_RATE']:
        assert getattr(cfg.MODEL, key) == getattr(reference.MODEL, key), key
    report = {'device': args.device, 'optimizer_steps': 0, 'checks': {'baseline_training_configuration': True}}
    # Keep baseline configuration checks separate from this small integration batch.
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.TEST.IMS_PER_BATCH = 8
    cfg.freeze()
    train_loader, _, val_loader, nq, nc, cams, views = make_dataloader(cfg)
    torch.manual_seed(1234)
    original = MLDCReID(cfg, nc, cams, views)
    # Compare initialized parameters and LR curves without storing a second large model on GPU.
    import hashlib
    expected = {n: hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
                for n, p in original.state_dict().items()}
    _, center = make_loss(cfg, nc)
    old_optimizer, _ = make_optimizer(cfg, original, center)
    configure_head_learning_rates(cfg, original, old_optimizer)
    old_scheduler = create_scheduler(cfg, old_optimizer)
    old_groups = {n: next(g for g in old_optimizer.param_groups if id(p) == id(g['params'][0]))
                  for n, p in original.named_parameters() if p.requires_grad}
    old_lr = {}
    for epoch in [0, 1, 19, 20, 60, 119, 120]:
        old_scheduler.step(epoch)
        old_lr[epoch] = {n: g['lr'] for n, g in old_groups.items()}
    del original, old_optimizer, old_scheduler, old_groups, center
    gc.collect()
    torch.manual_seed(1234)
    model = JointMLDCMUFASA(cfg, nc, cams, views)
    for n, p in model.mldc_model.state_dict().items():
        assert hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() == expected[n], n
    optimizer, scheduler = make_training(cfg, model, nc)
    groups = {n: next(g for g in optimizer.param_groups if id(p) == id(g['params'][0]))
              for n, p in model.named_parameters() if p.requires_grad}
    for epoch in old_lr:
        scheduler.step(epoch)
        for n, value in old_lr[epoch].items():
            assert abs(groups['mldc_model.'+n]['lr']-value) < 1e-15, (epoch, n)
    for n, g in groups.items():
        assert g['weight_decay'] == (cfg.SOLVER.WEIGHT_DECAY_BIAS if 'bias' in n else cfg.SOLVER.WEIGHT_DECAY)
    report['checks']['same_SSL_initialization_and_exact_global_LR_curve'] = True
    del expected, optimizer, scheduler, groups
    device = torch.device(args.device)
    model.to(device).train()
    assert all(block.training for block in model.mldc_model.base.blocks)
    assert model.mldc_model.bottleneck.training
    images, labels, camera, view = next(iter(train_loader))
    images, labels, camera, view = [t.to(device) for t in [images, labels, camera, view]]
    # Forward/backward does not update any parameter; snapshots are small selected tensors.
    before = model.mldc_model.base.blocks[0].attn.qkv.weight.detach().clone()
    with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
        parts = model(images, cam_label=camera, view_label=view)
        loss, terms = compute_loss(cfg, parts, labels)
    assert parts['feature'].shape == (8, 768)
    assert parts['maps'].shape == (8, 4, 128, 4)
    assert not parts['target'].requires_grad and parts['global_raw'].requires_grad
    torch.testing.assert_close(parts['fusion_scale'], torch.full((8,), .1, device=device))
    torch.testing.assert_close(parts['feature'].norm(dim=-1), torch.ones(8, device=device))
    loss.backward(retain_graph=True)
    active = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for n, p in active.items():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
    for prefix in ['mldc_model.base.blocks.0.', 'mldc_model.base.blocks.11.', 'mldc_model.mldc.',
                   'mldc_model.bottleneck.', 'mldc_model.classifier.', 'slots.encoders.0.',
                   'slots.encoders.3.', 'slots.projection.', 'readout.', 'gate.', 'classifier.', 'decoder.']:
        assert sum(float(p.grad.norm()) for n, p in active.items() if n.startswith(prefix)) > 0, prefix
    assert torch.equal(before, model.mldc_model.base.blocks[0].attn.qkv.weight)
    report['checks']['real_PK_AMP_backward_all_active_modules_no_parameter_update'] = True
    report['losses'] = {n: float(v.detach()) for n, v in terms.items()}
    model.zero_grad(set_to_none=True)
    # Isolate the residual route: its auxiliary loss must reach ViT patches/SA/readout.
    residual_metric_loss(cfg, parts, labels).backward()
    assert model.slots.encoders[0].value.weight.grad is not None
    assert model.mldc_model.base.blocks[8].attn.qkv.weight.grad is not None
    report['checks']['residual_identity_loss_reaches_backbone'] = True
    del parts, loss, terms
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        p1 = model(images, cam_label=camera, view_label=view, return_parts=True)
        p2 = model(images, cam_label=camera, view_label=view, return_parts=True)
    torch.testing.assert_close(p1['feature'], p2['feature'], atol=0, rtol=0)
    assert torch.equal(residual_feature(p1['global_feature'], p1['residual'], 0),
                       F.normalize(p1['global_feature'].float(), dim=-1))
    assert torch.all((p1['fusion_scale'] > 0) & (p1['fusion_scale'] < cfg.MUFASA.MAX_RESIDUAL_SCALE))
    assert 'fusion_scale' in diagnostic_values(p1)
    try:
        load_joint(model, {'kind': 'frozen_mldc_mufasa_v1'})
    except ValueError:
        pass
    else:
        raise AssertionError('Old increment was accepted')
    # Full model keys must include the backbone, BN buffers, gate and decoder.
    state = model.state_dict()
    assert any(k.startswith('mldc_model.base.blocks.') for k in state)
    assert 'mldc_model.bottleneck.num_batches_tracked' in state and 'gate.network.2.weight' in state
    load_joint(model, {'kind': CHECKPOINT_KIND, 'state_dict': state})
    report['checks']['deterministic_768D_output_full_state_and_old_checkpoint_rejection'] = True
    # Exercise CSV/evaluator/plot and fixed-pair visualization on a small existing-data subset.
    from torch.utils.data import DataLoader, Subset
    from mldc_mufasa_slots.visualization import plot_curves, save_visualizations
    from mldc_mufasa_slots.train import record_evaluation
    example = query_results(np.asarray([[0., .1, .2]]), [7, 7, 8, 7], [0, 0, 1, 1], 1,
                            ['q', 'same_camera', 'negative', 'positive'])
    assert example[0]['AP'] == .5
    records = val_loader.dataset.dataset
    query_indices = list(range(4))
    # The unchanged evaluator expects >=50 valid gallery positions after filtering.
    gallery_indices = list(range(nq, min(nq+128, len(records))))
    for j in query_indices:
        gallery_indices.append(next(i for i in range(nq, len(records))
                                    if records[i][1] == records[j][1] and records[i][2] != records[j][2]))
    gallery_indices = list(dict.fromkeys(gallery_indices))
    subset = Subset(val_loader.dataset, query_indices+gallery_indices)
    loader = DataLoader(subset, batch_size=8, collate_fn=val_loader.collate_fn)
    with tempfile.TemporaryDirectory(prefix='mufasa_joint_verify_') as temporary:
        check_cfg = cfg.clone()
        check_cfg.defrost()
        check_cfg.OUTPUT_DIR = temporary
        check_cfg.MUFASA.VIS_SAMPLES = 1
        check_cfg.freeze()
        result = evaluate(check_cfg, model, loader, len(query_indices), Path(temporary)/'evaluation')
        assert set(result['metrics']) == {'mldc', 'final', 'residual_only'}
        record_evaluation(check_cfg, 0, result)
        assert (Path(temporary)/'retrieval_curves.png').exists()
        save_visualizations(check_cfg, model, val_loader, nq, 0)
        assert any((Path(temporary)/'visualizations/epoch_000').glob('*.png'))
        optimizer, scheduler = make_training(cfg, model, nc)
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
        destination = Path(temporary)/'format_check.pth'
        save_checkpoint(cfg, model, optimizer, scheduler, scaler, 0, result, -1., destination)
        saved = torch.load(destination, map_location='cpu', weights_only=False)
        assert saved['kind'] == CHECKPOINT_KIND
        assert {'scheduler', 'scaler', 'optimizer', 'rng'} <= set(saved)
        assert 'mldc_model.bottleneck.num_batches_tracked' in saved['state_dict']
        load_joint(model, saved)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        report['checks']['full_joint_checkpoint_roundtrip_in_temporary_directory'] = True
        del saved, optimizer, scheduler, scaler
    report['checks']['current_MLDC_vs_final_evaluation_plot_and_fixed_pair_visualization'] = True
    report['trainable_parameters'] = sum(p.numel() for p in active.values())
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
