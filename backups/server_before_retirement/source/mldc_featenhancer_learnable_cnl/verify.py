"""Check the new feature path, CNL gradients, checkpoint and optional GPU capacity."""
import argparse
import gc
import json
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.nn import functional as F


def configuration():
    from config import cfg as base_cfg
    from .model.make_model_mldc import add_mldc_config
    cfg = add_mldc_config(base_cfg.clone())
    cfg.merge_from_file(str(Path(__file__).parent / 'configs/night600.yml'))
    return cfg


def cpu_checks():
    from mldc_learnable_global_cnl.model.make_model_mldc import MLDCReID as Baseline
    from .model.make_model_mldc import MLDCReID, VARIANT, configure_head_learning_rates
    from .global_cnl import GlobalCNL
    from .loss.make_loss_mldc import make_loss
    from .processor.processor_mldc import save_checkpoint
    from .scheduler_mldc import create_fixed_equal_scheduler, wrap_weight_scheduler
    from .verify_schedule import main as verify_schedule
    from solver import make_optimizer

    torch.set_num_threads(1)
    cfg = configuration();cfg.DATALOADER.NUM_WORKERS = 0;cfg.freeze()
    torch.manual_seed(1234)
    baseline = Baseline(cfg, 2, 2, 1).eval()
    torch.manual_seed(1234)
    model = MLDCReID(cfg, 2, 2, 1).eval()
    state = model.state_dict()
    assert all(torch.equal(tensor, state[name]) for name, tensor in baseline.state_dict().items())
    del state
    assert sum(p.numel() for p in model.feat_enhancer.parameters()) == 139400
    assert model.mldc.layer_weights.requires_grad and model.mldc.layer_weights.numel() == 4
    assert torch.equal(model.readout_weights(), torch.tensor([0.]*8+[.25]*4))
    images = torch.rand(4, 3, 256, 128) * 2 - 1
    with torch.no_grad():
        reference = baseline(images[:2])
        enhancer = model.feat_enhancer
        model.feat_enhancer = nn.Identity()
        assert torch.equal(reference, model(images[:2]))
        model.feat_enhancer = enhancer
        enhanced = enhancer(images[:2])
        captured = []
        handle = model.base.register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
        output = model(images[:2])
        handle.remove()
        assert len(captured) == 1 and torch.equal(captured[0], enhanced)
        assert output.shape == (2, 768) and torch.isfinite(output).all()
        assert enhanced.shape == images[:2].shape and not torch.equal(enhanced, images[:2])
    del baseline, reference, captured, enhanced, output;gc.collect()

    loss_fn, center = make_loss(cfg, 2)
    optimizer, _ = make_optimizer(cfg, model, center)
    configure_head_learning_rates(cfg, model, optimizer)
    scheduler = wrap_weight_scheduler(cfg, model, optimizer, create_fixed_equal_scheduler(cfg, optimizer))
    lr = {name: group['lr'] for group in optimizer.param_groups for name, parameter in model.named_parameters()
          if any(parameter is value for value in group['params'])}
    assert abs(lr['feat_enhancer.e_conv1.weight']-4e-6) < 1e-14
    assert abs(lr['feat_enhancer.e_conv1.bias']-8e-6) < 1e-14
    assert abs(lr['bottleneck.weight']-2e-5) < 1e-14
    assert abs(lr['mldc.layer_weights']-4e-6) < 1e-14
    model.train()
    labels = torch.tensor([0, 0, 1, 1]);cameras = torch.tensor([0, 1, 0, 1]);indices = torch.arange(4)
    scores, raw, unit = model(images, labels, return_cnl_features=True)
    assert scores.shape == (4, 2) and raw.shape == unit.shape == (4, 768)
    assert torch.allclose(unit.norm(dim=1), torch.ones(4), atol=1e-6)
    memory = GlobalCNL([0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 1, 0], dim=768)
    q = unit[0].detach()
    orthogonal = torch.randn_like(q);orthogonal -= orthogonal.dot(q)*q
    orthogonal = F.normalize(orthogonal, dim=0)
    memory.features[4] = q
    memory.features[5] = F.normalize(.96*q+(1-.96**2)**.5*orthogonal, dim=0)
    memory.valid[4:] = True
    before = {name: value.clone() for name, value in memory.state_dict().items()}
    cnl, stats = memory(unit, indices, labels, cameras)
    assert stats['active_anchors'] > 0 and cnl > 0 and torch.isfinite(cnl)
    selected = {'enhancer': model.feat_enhancer.e_conv1.weight,
                'fusion_logits': model.mldc.layer_weights,
                'patch_embed': model.base.patch_embed.proj.weight}
    gradients = torch.autograd.grad(cnl, list(selected.values()), retain_graph=True)
    cnl_norms = {name: float(g.float().norm()) for name, g in zip(selected, gradients)}
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients), cnl_norms
    original = loss_fn(scores, raw, labels, cameras)
    total = original + cfg.SOLVER.CNL_WEIGHT*cnl
    total.backward()
    missing = [name for name, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not missing, missing
    assert all(torch.equal(before[name], value) for name, value in memory.state_dict().items())
    terms = dict(total=float(total.detach()), cnl=float(cnl.detach()),
                 identity=float(loss_fn.last_terms['identity']),triplet=float(loss_fn.last_terms['triplet']))
    model.zero_grad(set_to_none=True)
    del scores, raw, unit, gradients, original, total, optimizer, scheduler, loss_fn, center;gc.collect()
    model.eval()
    with torch.no_grad():expected = model(images[:1])
    architecture = model.architecture()
    with tempfile.TemporaryDirectory(prefix='interface_check_', dir=str(Path(__file__).parent)) as folder:
        checkpoint = Path(folder)/'interface.pth'
        save_checkpoint(model, cfg, 0, checkpoint, memory=memory, dataset_fingerprint='generated-fixture')
        del model;gc.collect()
        restored_cfg = cfg.clone();restored_cfg.defrost()
        restored_cfg.MODEL.PRETRAIN_CHOICE = 'self';restored_cfg.MODEL.PRETRAIN_PATH = str(checkpoint)
        restored_cfg.freeze()
        restored = MLDCReID(restored_cfg, 2, 2, 1).eval()
        with torch.no_grad():assert torch.equal(expected, restored(images[:1]))
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert saved['variant'] == VARIANT and saved['architecture'] == architecture
        assert all(torch.equal(saved['cnl_memory'][name], value) for name, value in memory.state_dict().items())
        del restored, saved;gc.collect()
    schedule = verify_schedule()
    return dict(all_passed=True,architecture=architecture,enhancer_parameters=139400,
        original_backbone_and_head_initialization_identical=True,
        identity_enhancer_matches_original_output=True,normalized_input_enhancer_called_before_backbone=True,
        cpu_full_vit_forward_backward=True,cnl_only_gradient_norms=cnl_norms,
        all_active_parameter_gradients_finite=True,feature_dim=768,loss_terms=terms,
        checkpoint_and_memory_restore=True,schedule_checked=schedule['all_passed'],
        optimizer_steps=0,training_started=False)


def cuda_capacity(batch):
    from datasets import make_dataloader
    from solver import make_optimizer
    from .model.make_model_mldc import MLDCReID,configure_head_learning_rates
    from .loss.make_loss_mldc import make_loss
    from .indexed_data import indexed_loader
    from .global_cnl import GlobalCNL
    from .scheduler_mldc import create_fixed_equal_scheduler
    torch.set_num_threads(1)
    cfg = configuration();cfg.DATALOADER.NUM_WORKERS = 0;cfg.SOLVER.IMS_PER_BATCH = batch;cfg.freeze()
    torch.manual_seed(cfg.SOLVER.SEED)
    torch.backends.cudnn.deterministic = True;torch.backends.cudnn.benchmark = True
    assert torch.cuda.is_available()
    free, total_memory = torch.cuda.mem_get_info()
    if free < 14*1024**3:
        return dict(all_passed=False,resource_wait=True,free_bytes=free,
                    reason='Wait for at least 14 GiB free for the batch-64 capacity check',optimizer_steps=0)
    torch.cuda.reset_peak_memory_stats()
    loader, _, _, _, nc, cameras, views = make_dataloader(cfg)
    loader, dataset = indexed_loader(loader)
    images, labels, camera, view, indices = next(iter(loader))
    assert len(images) == batch
    model = MLDCReID(cfg, nc, cameras, views).cuda().train()
    loss_fn, center = make_loss(cfg, nc)
    optimizer, _ = make_optimizer(cfg, model, center)
    configure_head_learning_rates(cfg, model, optimizer)
    scheduler = create_fixed_equal_scheduler(cfg, optimizer)
    memory = GlobalCNL([int(r[1]) for r in dataset.entries],[int(r[2]) for r in dataset.entries],dim=768).cuda()
    images, labels, camera, view, indices = [value.cuda() for value in (images, labels, camera, view, indices)]
    scaler = torch.amp.GradScaler('cuda')
    finite = False
    for attempt in range(6):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            scores, raw, unit = model(images, labels, cam_label=camera, view_label=view, return_cnl_features=True)
            original = loss_fn(scores, raw, labels, camera)
            cnl, stats = memory(unit, indices, labels, camera)
            loss = original + cfg.SOLVER.CNL_WEIGHT*cnl
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        scaler.update()
        if finite:break
        del scores, raw, unit, original, cnl, loss
    torch.cuda.synchronize()
    assert finite, 'AMP gradients remained nonfinite after scale reduction'
    reserved = torch.cuda.max_memory_reserved()
    momentum_bytes = sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
    report = dict(all_passed=True,gpu_batch=batch,real_training_images=True,
        training_images=len(dataset),amp_attempts=attempt+1,finite_gradients=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=reserved,
        estimated_sgd_momentum_bytes=momentum_bytes,free_before_bytes=free,
        capacity_with_margin=reserved+momentum_bytes+1024**3<free,
        loss=float(loss.detach()),optimizer_steps=0,training_started=False)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda-batch',type=int)
    parser.add_argument('--report')
    args = parser.parse_args()
    report = cuda_capacity(args.cuda_batch) if args.cuda_batch else cpu_checks()
    if args.report:Path(args.report).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('VERIFICATION_RESULT '+json.dumps(report),flush=True)


if __name__=='__main__':main()
