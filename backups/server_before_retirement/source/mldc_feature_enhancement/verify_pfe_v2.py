"""CPU integration verification on real SSL + training images, no experiment run."""
import gc
import json
import random
import tempfile
import warnings
from pathlib import Path


def main():
    import numpy as np
    import torch
    from config import cfg as base_cfg
    from datasets import make_dataloader
    from .pfe_v2_model import add_config, config_from_dump, PFEV2ReID, RMSPFETokenAdapter
    from .pfe_v2_optim import build_optimizer, build_scheduler, lr_summary
    from .pfe_v2_diagnostics import probe_effect, render_probe
    from .model import objectives
    from .processor import save_checkpoint
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    cfg = add_config(base_cfg.clone()); cfg.merge_from_file(str(Path(__file__).with_name('variants/progressive_enhancement_norm_five_percent/config.yml')))
    cfg.DATALOADER.NUM_WORKERS = 0; cfg.SOLVER.IMS_PER_BATCH = 8; cfg.TEST.IMS_PER_BATCH = 8
    cfg.freeze()
    assert config_from_dump(base_cfg, cfg.dump()).dump() == cfg.dump()
    random.seed(cfg.SOLVER.SEED); np.random.seed(cfg.SOLVER.SEED); torch.manual_seed(cfg.SOLVER.SEED)
    train_loader, normal_loader, _, _, nc, cameras, views = make_dataloader(cfg)
    images, labels, camera, view = next(iter(train_loader))
    assert len(set(labels.tolist())) >= 2 and len(images) == 8
    # Direct AMP/zero-output checks on an inexpensive adapter.
    adapter = RMSPFETokenAdapter(input_dim=24, dim=8).train()
    test_input = torch.randn(4, 12, 24, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        changed, stats = adapter(test_input, (4,3), True)
        value = changed.square().mean()
    value.backward()
    assert changed.dtype == torch.float32
    assert torch.allclose(stats['correction_ratio'], torch.full((4,),.05), atol=1e-7, rtol=1e-5)
    assert test_input.grad is not None and torch.isfinite(test_input.grad).all()
    with torch.no_grad():
        adapter.up.weight.zero_()
        recovered, zero_stats = adapter(test_input, (4,3), True)
        assert torch.equal(recovered, test_input.float())
        assert not zero_stats['correction_ratio'].any()
    del adapter, test_input, changed, stats, value
    gc.collect()

    torch.manual_seed(cfg.SOLVER.SEED)
    model = PFEV2ReID(cfg, nc, cameras, views, initialize=True)
    assert all(p.requires_grad for p in model.pfe_adapters.parameters())
    assert all(p.requires_grad for b in model.base.blocks for p in b.parameters())
    clean_images, _, _, clean_cam, clean_view, _ = next(iter(normal_loader))
    summary, maps = probe_effect(model, clean_images[:3], clean_cam[:3], clean_view[:3])
    assert all(abs(r-.05) < 1e-6 for r in summary['stage_ratio_mean'])
    assert summary['descriptor_relative_change_mean'] > 1e-4
    report_dir = Path(cfg.OUTPUT_DIR).parent/'pfe_v2_preflight_20261018'
    render_probe(cfg, summary, maps, report_dir, 'startup_train')
    del maps
    optimizer = build_optimizer(cfg, model); scheduler = build_scheduler(cfg, optimizer)
    schedule = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for epoch in (0,1,5,20,80,119,120):
            scheduler.step(epoch)
            schedule[str(epoch)] = lr_summary(optimizer)
            for group in optimizer.param_groups:
                expected = optimizer.param_groups[0]['lr'] / optimizer.param_groups[0]['lr_multiplier']
                assert abs(group['lr']/group['lr_multiplier']-expected) <= max(1e-15, expected*1e-6)
        scheduler.step(20)  # exercise a real backward/update at high, not warmup LR
    originals = {n:p.detach().clone() for n,p in model.named_parameters() if n.startswith('pfe_adapters.')}
    model.train(); optimizer.zero_grad(set_to_none=True)
    # Actual FP32 full-network backward on PK training samples, no saved weights.
    data = model(images, cam_label=camera, view_label=view, diagnostics=True)
    assert torch.equal(data['raw_readout'], data['layers'][8:12].mean(0))
    assert torch.allclose(data['pfe_correction_ratio'], torch.full((8,4),.05), atol=1e-6)
    loss, identity, triplet = objectives(data, labels, cfg)
    assert torch.isfinite(loss) and torch.allclose(loss.detach(), identity+triplet)
    loss.backward()
    missing = [n for n,p in model.named_parameters() if n.startswith('pfe_adapters.') and
               (p.grad is None or not torch.isfinite(p.grad).all() or not p.grad.any())]
    assert not missing, missing
    gradients = {part: sum(float(p.grad.square().sum()) for n,p in model.named_parameters()
                    if part in n and p.grad is not None)**.5
                 for part in ('pfe_adapters.', 'base.patch_embed.', 'base.blocks.', 'classifier.')}
    assert all(v > 0 and np.isfinite(v) for v in gradients.values())
    optimizer.step()
    updates = {}
    for part in ('.down.', '.block.', '.up.'):
        old_sum = delta_sum = directional_sum = 0.
        for name,p in model.named_parameters():
            if name not in originals or part not in name: continue
            old = originals[name]; new = p.detach()
            scale = (old*new).sum()/old.square().sum().clamp_min(1e-30)
            old_sum += float(old.square().sum()); delta_sum += float((new-old).square().sum())
            directional_sum += float((new-scale*old).square().sum())
        updates[part] = dict(relative_total=(delta_sum/old_sum)**.5,
                            relative_direction=(directional_sum/old_sum)**.5)
        assert updates[part]['relative_direction'] > 1e-8
    del originals, data, loss, identity, triplet
    optimizer.zero_grad(set_to_none=True)
    # Verify checkpoint serialization uses a distinct variant and strict state layout.
    architecture = model.architecture()
    with tempfile.TemporaryDirectory(prefix='pfe_v2_interface_') as folder:
        checkpoint = Path(folder)/'interface.pth'
        save_checkpoint(checkpoint, model, cfg, 0, None)
        stored = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        assert stored['variant'] == architecture['variant']
        rebuilt_cfg = config_from_dump(base_cfg, stored['config']); rebuilt_cfg.freeze()
        clone = PFEV2ReID(rebuilt_cfg, nc, cameras, views, initialize=False)
        clone.load_state_dict(stored['state_dict'], strict=True, assign=True)
        assert clone.architecture() == architecture
        del stored, clone
    assert not torch.cuda.is_initialized()
    report = dict(architecture=architecture, startup_effect=summary, schedule=schedule,
        gradients=gradients, isolated_cpu_step_relative_updates=updates,
        real_ssl_loaded=True, samples='8 augmented TRAIN images for backward, PK=2x4; 3 clean TRAIN images for initial effect; no query/gallery inference',
        full_id_plus_triplet=True, joint_backbone_trainable=True,
        all_pfe_gradients_finite_nonzero=True, fp32_normalization_under_cpu_amp=True,
        zero_output_recovers_input_exactly=True, config_roundtrip_and_checkpoint_strict_load=True,
        full_network_isolated_cpu_optimizer_steps=1, experiment_started=False,
        cuda_initialized=False, figures_server=str(report_dir),
        note='Verification weights discarded. Initial effect is untrained and does not establish retrieval benefit.')
    (report_dir/'verification.json').write_text(json.dumps(report, indent=2))
    print('PFE_V2_VERIFIED', json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    main()
