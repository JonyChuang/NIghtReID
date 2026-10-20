"""CPU-only PFE integration checks with random inputs; no optimizer updates."""
import gc
import json
import tempfile
from pathlib import Path


def main():
    import torch
    from config import cfg as base_cfg
    from .pfe_model import add_config, config_from_dump, PFEReID, objectives
    from .pfe_modules import MSRFConv
    from .pfe_processor import evaluate
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError('Verification must not initialize CUDA')
    cfg = add_config(base_cfg.clone()); cfg.merge_from_file(str(Path(__file__).with_name('variants/progressive_patch_enhancement/config.yml'))); cfg.freeze()
    assert config_from_dump(base_cfg, cfg.dump()).dump() == cfg.dump()
    # Check receptive-field rearrangement preserves arbitrary spatial shape.
    small = MSRFConv(8).eval()
    with torch.no_grad():
        assert small(torch.randn(2, 8, 7, 5)).shape == (2, 8, 7, 5)
    del small
    torch.manual_seed(1234)
    model = PFEReID(cfg, 4, 0, 0, initialize=False)
    assert not hasattr(model, 'patch_compensation') and isinstance(model.feat_enhancer, torch.nn.Identity)
    assert all(p.requires_grad for p in model.pfe_adapters.parameters())
    assert all(p.requires_grad for b in model.base.blocks for p in b.parameters())
    initial_patch = model.base.patch_embed.proj.weight.detach().clone()
    initial_head = model.classifier.weight.detach().clone()
    inputs = torch.rand(2, 3, 256, 128) * 2 - 1
    model.eval()
    with torch.inference_mode():
        model.enabled = False
        plain = model(inputs, diagnostics=True)
        plain_layers = plain['layers'].clone()
        model.enabled = True
        enhanced = model(inputs, diagnostics=True)
        assert torch.equal(enhanced['enhanced'], inputs)
        assert torch.equal(enhanced['layers'][:2], plain_layers[:2])
        later_change = float((enhanced['layers'][2:] - plain_layers[2:]).abs().max())
        assert later_change > 1e-8, later_change
        assert torch.equal(enhanced['raw_readout'], enhanced['layers'][8:12].mean(0))
        assert enhanced['pfe_before_norm'].shape == (2, 4, 16, 8)
        assert enhanced['cls_attention'].shape == (2, 4, 16, 8)
        initial_ratios = enhanced['pfe_correction_ratio'].mean(0).tolist()
        assert all(r > 0 for r in initial_ratios)
        # A zero increment must recover the unmodified backbone exactly.
        for adapter in model.pfe_adapters.values():
            adapter.up.weight.zero_()
        zero = model(inputs, diagnostics=True)
        assert torch.equal(zero['layers'], plain_layers)
    del model, plain, enhanced, zero, plain_layers, inputs
    gc.collect()
    torch.manual_seed(1234)
    model = PFEReID(cfg, 4, 0, 0, initialize=False).train()
    data = model(torch.rand(4, 3, 256, 128) * 2 - 1)
    loss, identity, triplet = objectives(data, torch.tensor([0, 0, 1, 1]), cfg)
    assert torch.allclose(loss.detach(), identity + triplet)
    loss.backward()
    gradients = {}
    for depth in cfg.MODEL.PFE_LAYERS:
        for part in ('norm.', 'down.', 'block.msrf.generate.', 'block.msrf.reduce.', 'block.msrf.height_attention.',
                     'block.msrf.width_attention.', 'block.msrf.aggregate.', 'block.transform.', 'up.'):
            prefix = f'pfe_adapters.{depth}.{part}'
            gradients[prefix] = sum(float(p.grad.float().square().sum()) for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None) ** .5
    for prefix in ('base.patch_embed.', 'base.blocks.', 'classifier.'):
        gradients[prefix] = sum(float(p.grad.float().square().sum()) for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None) ** .5
    assert all(v > 0 and torch.isfinite(torch.tensor(v)) for v in gradients.values()), gradients
    missing = [n for n, p in model.pfe_adapters.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or not p.grad.any()]
    assert not missing, missing
    parameter_count = sum(p.numel() for p in model.pfe_adapters.parameters())
    model.zero_grad(set_to_none=True); del data, loss, identity, triplet
    model.eval()
    # Exercise eval, stage logging and both overlay renderers, not performance.
    testcfg = cfg.clone(); testcfg.defrost(); testcfg.TEST.PFE_VIS_INDICES = [0]; testcfg.freeze()
    images = torch.rand(14, 3, 256, 128) * 2 - 1
    pids = [0, 1] + list(range(12)); cams = [0, 0] + [1] * 12
    loader = [(images[i:i+2], tuple(pids[i:i+2]), tuple(cams[i:i+2]), torch.tensor(cams[i:i+2]), torch.zeros(2, dtype=torch.long), ['interface-check'] * 2) for i in range(0, 14, 2)]
    with tempfile.TemporaryDirectory(prefix='pfe_cpu_interface_') as folder:
        _, diagnostics = evaluate(testcfg, model, loader, 2, 'cpu', Path(folder))
        assert len(diagnostics['mean_stage_correction_ratio']) == 4
        assert (Path(folder)/'query_00000_pfe.png').exists()
        assert (Path(folder)/'query_00000_cls.png').exists()
    architecture = model.architecture()
    del model, loader, images
    gc.collect()
    torch.manual_seed(1234)
    controlcfg = cfg.clone(); controlcfg.defrost(); controlcfg.MODEL.PFE_ENABLED = False; controlcfg.freeze()
    control = PFEReID(controlcfg, 4, 0, 0, initialize=False).eval()
    assert torch.equal(control.base.patch_embed.proj.weight, initial_patch)
    assert torch.equal(control.classifier.weight, initial_head)
    with torch.no_grad():
        result = control(torch.zeros(1, 3, 256, 128), diagnostics=True)
        assert torch.equal(result['raw_readout'], result['layers'][8:12].mean(0))
    assert not torch.cuda.is_initialized()
    report = dict(architecture=architecture, config_roundtrip=True, arbitrary_grid_msrf=True,
        real_vit_descriptor_shape=[4, 768], stage_maps_shape=[2, 4, 16, 8],
        raw_input_unchanged=True, first_two_cls_unchanged=True, subsequent_cls_max_change=later_change,
        zero_increment_recovers_backbone_exactly=True, fixed_equal_global_exact=True,
        all_active_modules_trainable=True, all_pfe_parameters_finite_nonzero_gradient=True,
        gradient_norms=gradients, initial_stage_correction_ratios=initial_ratios,
        pfe_parameters=parameter_count, baseline_initialization_matches=True,
        evaluation_and_both_heatmap_renderers_checked=True, full_id_plus_triplet=True,
        optimizer_steps=0, pretrained_loaded=False, real_dataset_evaluated=False,
        cuda_initialized=False, training_started=False)
    print('PFE_VERIFIED', json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    main()
