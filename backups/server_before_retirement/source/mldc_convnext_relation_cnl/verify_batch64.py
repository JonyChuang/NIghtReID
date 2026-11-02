"""Actual batch64 AMP/ID/Triplet/CNL/KD preflight; zero optimizer steps.

Requires the user-supplied official teacher checkpoint and a visible GPU.
No download, training loop, checkpoint selection or performance tuning.
"""
import argparse
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='')
    args = parser.parse_args()
    import numpy as np
    import torch
    from torch.cuda import amp
    from config import cfg as defaults
    from datasets import make_dataloader
    from solver import make_optimizer
    from .indexed_data import indexed_loader
    from .model.make_model_mldc import add_mldc_config, make_model, configure_head_learning_rates
    from .loss.make_loss_mldc import make_loss
    from .global_cnl import GlobalCNL
    package = Path(__file__).resolve().parent
    cfg = add_mldc_config(defaults.clone())
    cfg.merge_from_file(str(package/'configs/convnext_relation_nightreid.yml'))
    if args.weights:
        cfg.PATCH_KD.TEACHER_PATH = args.weights
    cfg.freeze()
    if not Path(cfg.PATCH_KD.TEACHER_PATH).is_file():
        raise FileNotFoundError('Place the official teacher checkpoint at '+cfg.PATCH_KD.TEACHER_PATH)
    if not torch.cuda.is_available():
        raise RuntimeError('The full batch64 preflight requires a visible CUDA GPU')
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    loader, _, val, nq, nc, cameras, views = make_dataloader(cfg)
    loader, dataset = indexed_loader(loader)
    assert (len(dataset), nq, len(val.dataset)-nq, nc) == (15514, 1394, 26111, 500)
    model = make_model(cfg, nc, cameras, views).cuda().train()
    images, labels, camera, view, indices = [value.cuda() for value in next(iter(loader))]
    assert len(images) == 64
    loss_fn, center = make_loss(cfg, nc)
    optimizer, _ = make_optimizer(cfg, model, center)
    configure_head_learning_rates(cfg, model, optimizer)
    teacher = model.texture_teacher
    optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert not teacher.training and all(id(p) not in optimized for p in teacher.parameters())
    versions = {name: value._version for name, value in teacher.state_dict().items()}
    memory = GlobalCNL([int(row[1]) for row in dataset.entries], [int(row[2]) for row in dataset.entries],
        dim=768, gamma=cfg.SOLVER.CNL_GAMMA, temperature=cfg.SOLVER.CNL_TEMPERATURE,
        momentum=cfg.SOLVER.CNL_MOMENTUM).cuda()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    with amp.autocast():
        scores, raw, unit, terms = model(images, labels, cam_label=camera, view_label=view,
            return_cnl_features=True, return_distillation=True, sample_indices=indices)
        original = loss_fn(scores, raw, labels, camera)
        cnl, cnl_stats = memory(unit, indices, labels, camera)
        kd = model.distillation_loss(terms, cfg.PATCH_KD.RAMP_EPOCHS)
        combined = original + cfg.SOLVER.CNL_WEIGHT*cnl + kd
    assert torch.isfinite(combined) and kd > 0 and terms['stats']['images'] == 64
    assert terms['stats']['ordered_pairs_per_image'] == 16256
    assert terms['stats']['teacher_relation_std'] > 1e-6
    gradient = torch.autograd.grad(kd, model.base.blocks[-1].attn.qkv.weight, retain_graph=True)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    scaler = amp.GradScaler(init_scale=256.)
    scaler.scale(combined).backward()
    scaler.unscale_(optimizer)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(p.grad is None for p in teacher.parameters())
    assert all(value._version == versions[name] for name, value in teacher.state_dict().items())
    report = dict(all_passed=True, device=torch.cuda.get_device_name(0), seed=1234, real_batch=64,
        optimizer_steps=0, training_started=False, teacher_weights=cfg.PATCH_KD.TEACHER_PATH,
        teacher_frozen=True, teacher_in_optimizer=False, full_AMP_joint_backward_finite=True,
        original_loss=float(original.detach()), cnl_loss=float(cnl.detach()),
        kd_raw=float(terms['relation'].detach()), kd_weighted=float(kd.detach()),
        patch_statistics=terms['stats'], cnl_statistics=cnl_stats,
        gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        gpu_total_gib=torch.cuda.get_device_properties(0).total_memory/2**30)
    (package/'verification_batch64.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
