"""Real-image CPU checks, temporary modules only, zero optimizer updates."""
import ast
import gc
import json
from pathlib import Path
import tempfile
import torch
from torch.nn import functional as F
from config import cfg as defaults
from datasets import make_dataloader
from loss.triplet_loss import TripletLoss
from .model.base_model import MLDCReID as Baseline
from .model.make_model_mldc import MLDCReID,add_mldc_config,configure_head_learning_rates
from .model.texture_teacher import TextureTeacher,distillation_terms
from .export_inference import export
from .global_cnl import GlobalCNL


def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(1234)
    package=Path(__file__).resolve().parent
    # Compare the complete baseline protocol, allowing only auxiliary settings/output paths to vary.
    protocols={}
    for name in ['baseline','raw_feature','raw_structure','texture_feature','texture_structure']:
        item=add_mldc_config(defaults.clone())
        item.merge_from_file(str(package/'configs'/(name+'_nightreid.yml')))
        item=dict(item); item.pop('TEXTURE'); item.pop('OUTPUT_DIR')
        protocols[name]=item
    assert all(item==protocols['baseline'] for item in protocols.values())
    cfg=add_mldc_config(defaults.clone()); cfg.merge_from_file(str(package/'configs/texture_structure_nightreid.yml'))
    cfg.DATALOADER.NUM_WORKERS=0; cfg.TEST.IMS_PER_BATCH=4; cfg.freeze()
    train,_,val,nq,nc,cameras,views=make_dataloader(cfg)
    assert (len(train.dataset),nq,len(val.dataset)-nq,nc)==(15514,1394,26111,500)
    groups={}
    for row in train.dataset.dataset: groups.setdefault(int(row[1]),[]).append(row)
    entries=[]
    for pid in sorted(groups):
        group=groups[pid]; first=group[0]
        other=next((r for r in group if r[2]!=first[2]),None)
        if other is not None: entries.extend([first,other])
        if len(entries)==4: break
    small=type(val.dataset)(entries,val.dataset.transform)
    samples=[small[i] for i in range(4)]
    images=torch.stack([r[0] for r in samples]); labels=torch.tensor([r[1] for r in samples])
    camera=torch.tensor([r[2] for r in samples]); view=torch.tensor([r[3] for r in samples])
    stats={'five_configs_share_original_training_protocol':True}; rng={}
    root=package.parent
    # The forward methods of all original ViT primitives remain unchanged.
    original=ast.parse((root/'mldc_learnable_global_cnl/model/backbones/vit_pytorch_mldc.py').read_text())
    copied=ast.parse((package/'model/backbones/vit_pytorch_mldc.py').read_text())
    for name in ['Mlp','Attention','Block','PatchEmbed','TransReID']:
        a=next(n for n in original.body if isinstance(n,ast.ClassDef) and n.name==name)
        b=next(n for n in copied.body if isinstance(n,ast.ClassDef) and n.name==name)
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),name
    stats['original_vit_classes_unchanged']=True
    # Raw and texture teachers start with identical learned-head capacity and initialization.
    torch.manual_seed(1234); raw=TextureTeacher(nc,'raw')
    torch.manual_seed(1234); teacher=TextureTeacher(nc,'texture')
    assert sum(p.numel() for p in raw.parameters())==sum(p.numel() for p in teacher.parameters())
    assert all(torch.equal(p,dict(raw.named_parameters())[n]) for n,p in teacher.named_parameters())
    stats['matched_teacher_parameters']=sum(p.numel() for p in teacher.parameters())
    raw.initialize_stem(cfg.MODEL.PRETRAIN_PATH); raw.train()
    raw_output=raw(images)
    raw_loss=F.cross_entropy(raw_output['logits'],labels)+TripletLoss(.3)(raw_output['raw'],labels)[0]
    raw_loss.backward()
    assert raw.encoder[0].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in raw.stem.parameters())
    stats['raw_teacher_identity_backward']=True
    del raw,raw_output,raw_loss
    probe=torch.randn(2,128,16,8,requires_grad=True); target=torch.randn_like(probe)
    feature=distillation_terms(probe,target)['feature']; feature.backward()
    assert torch.isfinite(probe.grad).all() and probe.grad.abs().sum()>0
    stats['feature_control_backward']=True
    del probe,target,feature
    teacher.initialize_stem(cfg.MODEL.PRETRAIN_PATH); teacher.train()
    stem_before={n:t.clone() for n,t in teacher.stem.state_dict().items()}
    teacher_output=teacher(images)
    teacher_loss=F.cross_entropy(teacher_output['logits'],labels)+TripletLoss(.3)(teacher_output['raw'],labels)[0]
    teacher_loss.backward()
    assert all(p.grad is None for p in teacher.stem.parameters())
    assert teacher.encoder[0].weight.grad.abs().sum()>0
    assert teacher.classifier.weight.grad.abs().sum()>0
    assert all(torch.equal(v,stem_before[n]) for n,v in teacher.stem.state_dict().items())
    stats['teacher_identity_gradient']=float(teacher.encoder[0].weight.grad.norm())
    stats['teacher_stem_parameters_and_BN_frozen']=True
    teacher.zero_grad(set_to_none=True); teacher.eval()
    # Refuse to turn the untrained verification fixture into a real training teacher.
    with tempfile.TemporaryDirectory(prefix='texture_teacher_fixture_') as temp:
        bad=Path(temp)/'untrained.pth'
        torch.save(dict(state_dict=teacher.state_dict(),teacher_metadata=dict(trained=False,bank='texture',seed=1234)),bad)
        try: teacher.load_trained(bad)
        except (ValueError,KeyError): stats['untrained_teacher_rejected']=True
        else: raise AssertionError('Untrained teacher accepted')
    # Compute baseline initial outputs, then release the full baseline before student construction.
    torch.manual_seed(1234); baseline=Baseline(cfg,nc,cameras,views).eval()
    rng['baseline']=torch.get_rng_state().clone()
    with torch.inference_mode(): expected,expected_raw,expected_layers=baseline(images[:2],return_components=True)
    trained_path=Path('/root/log/transreid/nightreid/mldc_learnable_global_cnl/deep_four_uniform_cross_camera_cnl_fixed_equal_schedule/transformer_best.pth')
    original_stat=trained_path.stat()
    baseline.load_param(trained_path)
    with torch.inference_mode(): expected_trained=baseline(images[:2])
    del baseline; gc.collect()
    torch.manual_seed(1234); model=MLDCReID(cfg,nc,cameras,views,teacher_override=teacher).eval()
    assert torch.equal(torch.get_rng_state(),rng['baseline'])
    calls=[]
    hook=model.texture_teacher.register_forward_hook(lambda *args:calls.append(1))
    with torch.inference_mode(): actual,raw,layers=model(images[:2],return_components=True)
    assert not calls
    assert torch.equal(actual,expected) and torch.equal(raw,expected_raw) and torch.equal(layers,expected_layers)
    stats['initial_SSL_all_CLS_and_descriptor_exact']=True
    stats['initial_max_error']=float((actual-expected).abs().max())
    checkpoint=torch.load(trained_path,map_location='cpu',mmap=True,weights_only=False)
    incompatible=model.load_state_dict(checkpoint['state_dict'],strict=False)
    assert not incompatible.unexpected_keys
    assert all(k.startswith(('texture_teacher.','texture_adapter.')) for k in incompatible.missing_keys)
    del checkpoint; gc.collect()
    with torch.inference_mode(): actual_trained=model(images[:2])
    assert torch.equal(actual_trained,expected_trained)
    stats['trained_baseline_output_exact']=True
    # All active backbone blocks are trainable by default; the CPU check below narrows gradients only.
    assert all(p.requires_grad for block in model.base.blocks for p in block.parameters())
    assert all(p.requires_grad for p in model.base.patch_embed.parameters())
    stats['default_all_active_backbone_parameters_trainable']=True
    model.texture_enabled=False; model.train(); before_calls=len(calls)
    with torch.no_grad():
        *_,disabled=model(images,labels,cam_label=camera,view_label=view,
            return_cnl_features=True,return_distillation=True)
    assert len(calls)==before_calls and all(float(t)==0 for t in disabled.values())
    model.texture_enabled=True
    stats['disabled_baseline_training_has_zero_KD_and_no_teacher_calls']=True
    # Auxiliary gradients must reach the selected ViT block; teachers remain frozen.
    model.train(); buffers={n:t.clone() for n,t in model.texture_teacher.named_buffers()}
    model.base.requires_grad_(False); model.base.blocks[3].requires_grad_(True)
    scores,features,unit,terms=model(images,labels,cam_label=camera,view_label=view,
        return_cnl_features=True,return_distillation=True)
    cnl=GlobalCNL(labels.tolist(),camera.tolist(),dim=768,gamma=.9,temperature=.05,momentum=.2)
    index=torch.arange(4); cnl.update(unit.detach(),index,labels,camera)
    cnl_loss,_=cnl(unit,index,labels,camera)
    base_loss=F.cross_entropy(scores,labels)+TripletLoss(.3)(features,labels)[0]+.1*cnl_loss
    kd=model.distillation_loss(terms,20)
    assert torch.isfinite(kd) and kd>0
    kd.backward(retain_graph=True)
    selected_grad=float(model.base.blocks[3].attn.qkv.weight.grad.norm())
    assert selected_grad>0 and model.texture_adapter[1].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.texture_teacher.parameters())
    assert all(torch.equal(t,buffers[n]) for n,t in model.texture_teacher.named_buffers())
    stats['KD_gradient_selected_block']=selected_grad
    stats['teacher_frozen_during_student_training']=True
    model.zero_grad(set_to_none=True); (base_loss+kd).backward()
    assert torch.isfinite(model.base.blocks[3].attn.qkv.weight.grad).all()
    stats['combined_ID_Triplet_CNL_KD_backward']=True
    stats['loss_terms']={n:float(t.detach()) for n,t in terms.items()}
    stats['weighted_KD']=float(kd.detach())
    model.zero_grad(set_to_none=True)
    del scores,features,unit,terms,base_loss,kd,cnl,cnl_loss; gc.collect()
    from solver import make_optimizer
    from .loss.make_loss_mldc import make_loss
    loss_fn,center=make_loss(cfg,nc); optimizer,_=make_optimizer(cfg,model,center)
    configure_head_learning_rates(cfg,model,optimizer)
    covered={id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(id(p) in covered for p in model.texture_adapter.parameters())
    assert all(id(p) not in covered for p in model.texture_teacher.parameters())
    stats['optimizer_excludes_teacher_and_covers_adapter']=True
    del optimizer,center,loss_fn
    hook.remove(); model.eval()
    with torch.inference_mode(): restored_expected=model(images[:2])
    with tempfile.TemporaryDirectory(prefix='texture_student_roundtrip_') as temp:
        full=Path(temp)/'full.pth'; core=Path(temp)/'core.pth'
        torch.save(dict(state_dict=model.state_dict(),architecture=model.architecture(),config=cfg.dump(),
            epoch=0,mldc_layers=[9,10,11,12],temperature=.9,train_feature='mldc',readout='mldc',
            teacher_metadata={'trained':False,'verification_fixture':True}),full)
        export(full,core)
        del model,teacher; gc.collect()
        restored=MLDCReID(cfg,nc,cameras,views,initialize=False).eval(); restored.load_param(full)
        with torch.inference_mode(): out=restored(images[:2])
        assert torch.equal(out,restored_expected)
        del restored; gc.collect()
        original=Baseline(cfg,nc,cameras,views,initialize=False).eval(); original.load_param(core)
        with torch.inference_mode(): out=original(images[:2])
        assert torch.equal(out,restored_expected)
        stats['strict_student_checkpoint_roundtrip']=True
        stats['inference_export_loads_original_baseline_exact']=True
        del original; gc.collect()
    after=trained_path.stat()
    assert (after.st_size,after.st_mtime_ns)==(original_stat.st_size,original_stat.st_mtime_ns)
    stats.update(all_passed=True,seed=1234,device='cpu',dataset='nightreid',real_images=4,
        optimizer_steps=0,training_started=False,new_teacher_trained=False,
        performance_measured=False,CUDA_capacity_tested=False,original_checkpoint_unchanged=True)
    stats['CPU_backward_scope']='Only block4 and auxiliary/head parameters; production defaults keep all active backbone parameters trainable'
    outdir=root/'analysis_texture_distill_verification_20261026'; outdir.mkdir(exist_ok=True)
    (outdir/'verification.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print('VERIFICATION '+json.dumps(stats),flush=True)


if __name__=='__main__': main()
