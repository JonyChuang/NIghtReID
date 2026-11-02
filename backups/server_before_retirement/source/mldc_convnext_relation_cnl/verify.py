"""CPU/CUDA integration verification with zero optimizer updates.

Without --weights, random official-shape fixtures check wiring only. This is
explicitly not a pretrained model, NightReID evaluation or performance evidence.
"""
import argparse
import ast
import gc
import json
from pathlib import Path
import random
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository-root', default='.')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--weights', default='')
    parser.add_argument('--output', default='')
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    sys.path.insert(0, str(Path(args.repository_root).resolve()))
    sys.path.insert(0, str(package.parent))
    import numpy as np
    import torch
    import yaml
    from config.defaults import _C as defaults
    from mldc_convnext_relation_cnl.model.base_model import MLDCReID as Baseline
    from mldc_convnext_relation_cnl.model.make_model_mldc import add_mldc_config, MLDCReID
    from mldc_convnext_relation_cnl.model.patch_relation import patch_relation_loss
    from mldc_convnext_relation_cnl.model.convnext_teacher import FrozenConvNeXtTeacher
    from mldc_convnext_relation_cnl.scheduler_mldc import create_fixed_equal_scheduler
    torch.set_num_threads(2)
    device = torch.device(args.device)

    def seed():
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1234)

    cfg = add_mldc_config(defaults.clone())
    cfg.merge_from_file(str(package/'configs/convnext_relation_nightreid.yml'))
    cfg.MODEL.DEVICE = args.device
    if args.weights:
        cfg.PATCH_KD.TEACHER_PATH = args.weights
    cfg.freeze()
    report = dict(seed=1234, device=args.device, optimizer_steps=0, training_started=False,
        official_pretrained_weights_tested=bool(args.weights),
        fixture_note='Random initialization checks wiring only' if not args.weights else 'Local supplied weights',
        checkpoint_missing_behavior='raise FileNotFoundError, never auto-download')
    for path in package.rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    raw_source = (package/'model/convnext_official/upstream_convnext.py.txt').read_text(encoding='utf-8')
    adapted = (package/'model/convnext_official/convnext.py').read_text(encoding='utf-8')
    assert adapted == raw_source.replace(
        'from timm.models.layers import trunc_normal_, DropPath\nfrom timm.models.registry import register_model',
        'from .compat import trunc_normal_, DropPath, register_model')
    report['official_source_only_dependency_imports_changed'] = True
    config = yaml.safe_load((package/'configs/convnext_relation_nightreid.yml').read_text(encoding='utf-8'))
    control = yaml.safe_load((package/'configs/baseline_nightreid.yml').read_text(encoding='utf-8'))
    for item in (config, control):
        item.pop('PATCH_KD')
        item.pop('OUTPUT_DIR')
    assert config == control
    assert cfg.SOLVER.SEED == 1234 and cfg.SOLVER.MAX_EPOCHS == 120 and cfg.SOLVER.IMS_PER_BATCH == 64
    assert cfg.SOLVER.WARMUP_EPOCHS == 20 and cfg.SOLVER.WARMUP_METHOD == 'cosine'
    dummy = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([dummy], lr=cfg.SOLVER.BASE_LR)
    scheduler = create_fixed_equal_scheduler(cfg, optimizer)
    report['scheduler_factors'] = {str(epoch): scheduler.lr_lambdas[0](epoch)
                                   for epoch in [0, 1, 19, 20, 60, 119, 120]}
    report['baseline_control_protocol_matches'] = True

    # Independently computable relation fixture: four errors among six ordered pairs.
    s = torch.tensor([[[1., 0.], [1., 0.], [0., 1.]]], requires_grad=True)
    t = torch.tensor([[[1., 0.], [0., 1.], [0., 1.]]], requires_grad=True)
    loss, stats = patch_relation_loss(s, t)
    assert torch.allclose(loss, torch.tensor(2./3.))
    loss.backward()
    assert s.grad.norm() > 0 and t.grad is None and stats['ordered_pairs_per_image'] == 6
    equal, _ = patch_relation_loss(s.detach(), torch.nn.functional.pad(s.detach(), (0, 3)))
    assert equal == 0
    scaled, _ = patch_relation_loss(s.detach()*3., t.detach()*7.)
    assert torch.allclose(loss, scaled)
    try:
        patch_relation_loss(s, t[:, :2])
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched spatial grids must be rejected')
    report['relation_closed_form_channel_independence_and_stop_gradient'] = True

    images = (torch.rand(2, 3, 256, 128, generator=torch.Generator().manual_seed(1234))*2-1).to(device)
    with tempfile.TemporaryDirectory(prefix='convnext_relation_verify_') as directory:
        fixture_path = Path(directory)/'student.pth'
        seed()
        baseline = Baseline(cfg, 500, 6, 1, initialize=False).to(device).eval()
        expected_rng = torch.get_rng_state().clone()
        with torch.no_grad():
            expected = [v.cpu() for v in baseline(images, return_components=True)]
        torch.save(baseline.state_dict(), fixture_path)
        del baseline
        gc.collect()
        seed()
        model = MLDCReID(cfg, 500, 6, 1, initialize=False)
        if args.weights:
            with torch.random.fork_rng(devices=[]):
                model.texture_teacher = FrozenConvNeXtTeacher(args.weights, load_weights=True)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        reference = torch.load(fixture_path, mmap=True, weights_only=True)
        actual_core = model.inference_state_dict()
        assert actual_core.keys() == reference.keys()
        assert all(torch.equal(value, reference[key]) for key, value in actual_core.items())
        del reference, actual_core
        model.to(device).eval()
        calls = []
        hook = model.texture_teacher.register_forward_hook(lambda *items: calls.append(1))
        with torch.no_grad():
            actual = model(images, return_components=True)
        errors = [float((a.cpu()-b).abs().max()) for a, b in zip(actual, expected)]
        assert errors == [0., 0., 0.] and not calls
        report.update(initial_student_state_matches_baseline=True, student_RNG_unchanged=True,
                      initial_descriptor_raw_and_CLS_errors=errors, inference_teacher_calls=0)
        teacher = model.texture_teacher
        assert sum(p.numel() for p in teacher.network.parameters()) == 28589128
        normalized = teacher.normalize_input(images)
        expected_rgb = (images.float()*.5+.5-torch.tensor([.485,.456,.406],device=device)[None,:,None,None])
        expected_rgb = expected_rgb/torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
        assert torch.equal(normalized, expected_rgb)
        report['imagenet_normalization_same_geometry'] = True
        model.train()
        assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())
        versions = {name: value._version for name, value in teacher.state_dict().items()}
        active = [p for p in model.parameters() if p.requires_grad]
        optimized_ids = {id(p) for p in active}
        assert all(id(p) not in optimized_ids for p in teacher.parameters())
        optimizer = torch.optim.SGD(active, lr=.0004, momentum=.9)
        optimizer.zero_grad(set_to_none=True)
        labels = torch.tensor([0, 1], device=device)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            score, raw, unit, terms = model(images, labels, return_cnl_features=True,
                return_distillation=True, sample_indices=torch.arange(2, device=device))
            kd = model.distillation_loss(terms, 20)
            combined = torch.nn.functional.cross_entropy(score.float(), labels) + kd
        assert torch.isfinite(combined) and kd > 0 and terms['stats']['patches'] == 128
        targets = {'first_block': model.base.blocks[0].attn.qkv.weight,
                   'last_block': model.base.blocks[-1].attn.qkv.weight,
                   'original_stem': next(model.base.patch_embed.parameters())}
        gradients = torch.autograd.grad(kd, list(targets.values()), retain_graph=True)
        norms = {key: float(value.float().norm()) for key, value in zip(targets, gradients)}
        assert all(np.isfinite(value) and value > 0 for value in norms.values())
        combined.backward()
        assert all(torch.isfinite(p.grad).all() for p in active if p.grad is not None)
        assert all(p.grad is None for p in teacher.parameters())
        assert all(value._version == versions[name] for name, value in teacher.state_dict().items())
        assert not teacher.training and len(calls) == 1
        assert not model.base.blocks[-1]._forward_hooks  # capture handle removed
        assert all(p.requires_grad for block in model.base.blocks for p in block.parameters())
        report.update(teacher_parameters=28589128, teacher_frozen=True, teacher_in_optimizer=False,
            teacher_weights_unchanged=True, kd_gradients=norms, kd_raw=float(terms['relation'].detach()),
            kd_weighted=float(kd.detach()), relation_statistics=terms['stats'],
            full_student_CE_plus_KD_backward_finite=True,
            note='ID/Triplet/CNL code and schedule retained; this fixture checks the new teacher/KD wiring')
        hook.remove()
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            expected_reload = model(images).cpu()
        torch.save(dict(state_dict=model.inference_state_dict(), epoch=0,
                        architecture=model.architecture()), fixture_path)
        # Drop graph/optimizer references before loading the inference-only model.
        del optimizer, active, targets, gradients, score, raw, unit, terms, kd, combined, teacher, model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        reload_cfg = cfg.clone()
        reload_cfg.defrost()
        reload_cfg.MODEL.PRETRAIN_CHOICE = 'self'
        reload_cfg.MODEL.PRETRAIN_PATH = str(fixture_path)
        reload_cfg.PATCH_KD.TEACHER_PATH = '__missing_teacher_is_not_needed_at_inference__.pth'
        reload_cfg.freeze()
        restored = MLDCReID(reload_cfg, 500, 6, 1, initialize=True).to(device).eval()
        assert restored.texture_teacher is None
        with torch.no_grad():
            reload_error = float((restored(images).cpu()-expected_reload).abs().max())
        assert reload_error == 0.
        report.update(teacher_free_checkpoint_roundtrip_error=reload_error,
                      teacher_file_not_needed_for_inference=True)
    report['all_passed'] = True
    destination = Path(args.output) if args.output else package/'verification.json'
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
