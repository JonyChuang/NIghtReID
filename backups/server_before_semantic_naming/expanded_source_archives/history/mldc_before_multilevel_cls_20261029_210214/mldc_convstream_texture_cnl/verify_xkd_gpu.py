"""Real-batch CUDA preflight, with zero optimizer steps and no test tuning."""
import gc
import json
from pathlib import Path
import random


def main():
    import numpy as np
    import torch
    from torch.cuda import amp
    from config import cfg as defaults
    from datasets import make_dataloader
    from solver import make_optimizer
    from .indexed_data import indexed_loader
    from .model.base_model import MLDCReID as Baseline
    from .model.make_model_mldc import add_mldc_config,make_model,configure_head_learning_rates
    from .model.final_contrast import FinalContrast
    from .loss.make_loss_mldc import make_loss
    from .global_cnl import GlobalCNL
    package=Path(__file__).resolve().parent
    cfg=add_mldc_config(defaults.clone()); cfg.merge_from_file(str(package/'configs/texture_xkd_nightreid.yml'))
    control=add_mldc_config(defaults.clone()); control.merge_from_file(str(package/'configs/baseline_nightreid.yml'))
    def protocol(item):
        item=dict(item); item.pop('TEXTURE'); item.pop('OUTPUT_DIR'); return item
    assert protocol(cfg)==protocol(control)
    assert cfg.SOLVER.SEED==1234 and cfg.TEXTURE.OBJECTIVE=='identity_infonce'
    assert torch.cuda.is_available()
    cfg.freeze(); device=torch.device('cuda:0')
    def seed():
        random.seed(1234); np.random.seed(1234); torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
    seed(); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=True
    train,_,val,nq,nc,cameras,views=make_dataloader(cfg); train,dataset=indexed_loader(train)
    assert (len(dataset),nq,len(val.dataset)-nq,nc)==(15514,1394,26111,500)
    batch=next(iter(train)); images,labels,camera,view,indices=[x.to(device) for x in batch]
    assert len(images)==64
    report=dict(seed=1234,device=torch.cuda.get_device_name(0),real_training_batch=len(images),
        optimizer_steps=0,protocol_matches_baseline=True,objective=cfg.TEXTURE.OBJECTIVE)
    # Verify unchanged SSL initialization, all CLS layers, and original RNG stream.
    seed(); baseline=Baseline(cfg,nc,cameras,views).to(device).eval()
    expected_rng=torch.get_rng_state(); core_keys=set(baseline.state_dict())
    with torch.inference_mode(): expected=[v.cpu() for v in baseline(images[:2],return_components=True)]
    del baseline; gc.collect(); torch.cuda.empty_cache()
    seed(); model=make_model(cfg,nc,cameras,views)
    assert torch.equal(torch.get_rng_state(),expected_rng)
    assert model.texture_teacher.provenance['training_dataset_fingerprint']==dataset.fingerprint
    model.to(device).eval(); calls=[]
    hook=model.texture_teacher.convstream.register_forward_hook(lambda *args:calls.append(1))
    with torch.inference_mode(): actual=model(images[:2],return_components=True)
    errors=[float((a.cpu()-b).abs().max()) for a,b in zip(actual,expected)]
    assert errors==[0.,0.,0.] and not calls
    assert set(model.inference_state_dict())==core_keys
    assert all(p.requires_grad for block in model.base.blocks for p in block.parameters())
    assert all(p.requires_grad for p in model.base.patch_embed.parameters())
    report.update(initial_descriptor_raw_and_all_CLS_errors=errors,baseline_RNG_exact=True,
        inference_teacher_calls=0,inference_state_keys_match_original=True,entire_active_backbone_trainable=True)
    del actual,expected; gc.collect(); torch.cuda.empty_cache()
    # Independent loss semantics: remove same-ID negatives, preserve teacher stop-gradient.
    fixture=FinalContrast().to(device)
    u=torch.randn(6,768,device=device,requires_grad=True); t=torch.randn(6,256,device=device,requires_grad=True)
    ids=torch.tensor([0,0,1,1,2,2],device=device); cams=torch.tensor([0,1,0,1,0,1],device=device)
    loss,details=fixture(u,t,ids,cams,torch.arange(6,device=device)); loss.backward()
    assert torch.isfinite(loss) and loss>0 and t.grad is None and u.grad.norm()>0
    assert details['same_identity_negatives']==0 and details['negative_pairs']==24 and details['positive_pairs']==6
    assert fixture.student.weight.grad.norm()>0 and fixture.teacher.weight.grad.norm()>0
    zero,zero_stats=fixture(u,t,torch.zeros_like(ids),cams,torch.arange(6,device=device))
    assert zero==0 and zero_stats['active_anchors']==0
    report['InfoNCE_identity_masks_and_stop_gradient_passed']=True
    del fixture,u,t,loss,zero; gc.collect(); torch.cuda.empty_cache()
    # Exactly the production AMP + ID + Triplet + CNL + KD path, full batch and full backbone.
    model.train(); teacher_before={name:v.detach().cpu().clone() for name,v in model.texture_teacher.state_dict().items()}
    loss_fn,center=make_loss(cfg,nc); optimizer,_=make_optimizer(cfg,model,center)
    configure_head_learning_rates(cfg,model,optimizer)
    optimized={id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(id(p) not in optimized for p in model.texture_teacher.parameters())
    for group in optimizer.param_groups:
        if any(id(p)==id(model.texture_contrast.student.weight) for p in group['params']):
            assert abs(group['lr']-cfg.SOLVER.BASE_LR*cfg.TEXTURE.PROJECTION_LR_FACTOR)<1e-12
    memory=GlobalCNL([int(r[1]) for r in dataset.entries],[int(r[2]) for r in dataset.entries],
        dim=768,gamma=cfg.SOLVER.CNL_GAMMA,temperature=cfg.SOLVER.CNL_TEMPERATURE,
        momentum=cfg.SOLVER.CNL_MOMENTUM).to(device)
    torch.cuda.reset_peak_memory_stats()
    with amp.autocast():
        score,raw,unit,terms=model(images,labels,cam_label=camera,view_label=view,
            return_cnl_features=True,return_distillation=True,sample_indices=indices)
        original=loss_fn(score,raw,labels,camera)
        cnl,cnl_stats=memory(unit,indices,labels,camera)
        kd=model.distillation_loss(terms,cfg.TEXTURE.RAMP_EPOCHS)
        combined=original+cfg.SOLVER.CNL_WEIGHT*cnl+kd
    assert torch.isfinite(combined) and kd>0
    assert terms['stats']['active_anchors']==64 and terms['stats']['same_identity_negatives']==0
    assert terms['stats']['teacher_embedding_variance']>1e-8
    targets={'final_unit':unit,'BNneck':model.bottleneck.weight,'MLDC':model.mldc.layer_weights,
        'last_ViT_attention':model.base.blocks[-1].attn.qkv.weight,
        'first_ViT_attention':model.base.blocks[0].attn.qkv.weight,
        'student_projection':model.texture_contrast.student.weight,
        'teacher_projection':model.texture_contrast.teacher.weight}
    gradients=torch.autograd.grad(kd,list(targets.values()),retain_graph=True)
    norms={name:float(g.detach().float().norm()) for name,g in zip(targets,gradients)}
    assert all(np.isfinite(v) and v>0 for v in norms.values())
    del gradients
    # A small verification scale avoids false FP16 overflow from an uncalibrated default scaler.
    scaler=amp.GradScaler(init_scale=256.)
    scaler.scale(combined).backward(); scaler.unscale_(optimizer)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(p.grad is None for p in model.texture_teacher.parameters())
    assert all(torch.equal(v.cpu(),teacher_before[name]) for name,v in model.texture_teacher.state_dict().items())
    assert not model.texture_teacher.training and len(calls)==1
    peak=torch.cuda.max_memory_allocated()/2**30; total=torch.cuda.get_device_properties(0).total_memory/2**30
    assert peak<total*.9
    report.update(all_passed=True,teacher_checkpoint=cfg.TEXTURE.TEACHER_PATH,
        teacher_metadata=model.texture_teacher.provenance,teacher_frozen_and_excluded_from_optimizer=True,
        final_KD_gradient_norms=norms,original_loss=float(original.detach()),CNL_loss=float(cnl.detach()),
        KD_raw_loss=float(terms['relation'].detach()),KD_weighted_loss=float(kd.detach()),
        InfoNCE=terms['stats'],CNL=cnl_stats,gpu_peak_allocated_gib=peak,gpu_total_gib=total,
        full_batch_AMP_backward_passed=True,all_gradients_finite=True,projection_LR_factor=cfg.TEXTURE.PROJECTION_LR_FACTOR)
    hook.remove()
    (package/'verification_xkd_gpu.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('XKD_GPU_VERIFICATION '+json.dumps(report),flush=True)


if __name__=='__main__': main()
