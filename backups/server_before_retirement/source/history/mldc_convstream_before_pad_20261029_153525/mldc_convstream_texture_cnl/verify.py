"""Zero-update real-image and controlled-target checks of the changed paths."""
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
from .model.texture_teacher import TextureTeacher,exposure_view,specialization_consistency
from .model.final_relation import local_pair_scores,relative_distillation
from .export_inference import export
from .global_cnl import GlobalCNL


def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(1234)
    package=Path(__file__).resolve().parent; root=package.parent
    protocols=[]
    for name in ['baseline','raw_relation','texture_relation']:
        item=add_mldc_config(defaults.clone()); item.merge_from_file(str(package/'configs'/(name+'_nightreid.yml')))
        item=dict(item); item.pop('TEXTURE'); item.pop('OUTPUT_DIR'); protocols.append(item)
    assert all(p==protocols[0] for p in protocols)
    cfg=add_mldc_config(defaults.clone()); cfg.merge_from_file(str(package/'configs/texture_relation_nightreid.yml'))
    cfg.DATALOADER.NUM_WORKERS=0; cfg.TEST.IMS_PER_BATCH=4; cfg.freeze()
    train,_,val,nq,nc,cameras,views=make_dataloader(cfg)
    assert (len(train.dataset),nq,len(val.dataset)-nq,nc)==(15514,1394,26111,500)
    groups={}
    for row in train.dataset.dataset: groups.setdefault(int(row[1]),[]).append(row)
    entries=[]
    for pid in sorted(groups):
        group=groups[pid]; first=group[0]; other=next((r for r in group if r[2]!=first[2]),None)
        if other is not None: entries.extend([first,other])
        if len(entries)==4: break
    small=type(val.dataset)(entries,val.dataset.transform); samples=[small[i] for i in range(4)]
    images=torch.stack([r[0] for r in samples]); labels=torch.tensor([r[1] for r in samples])
    camera=torch.tensor([r[2] for r in samples]); view=torch.tensor([r[3] for r in samples]); indices=torch.arange(4)
    stats={'three_configs_share_original_student_protocol':True}
    def parsed(path): return ast.parse(path.read_text(encoding='utf-8'))
    original=parsed(root/'mldc_learnable_global_cnl/model/backbones/vit_pytorch_mldc.py')
    copied=parsed(package/'model/backbones/vit_pytorch_mldc.py')
    for name in ['Mlp','Attention','Block','PatchEmbed','TransReID']:
        a=next(n for n in original.body if isinstance(n,ast.ClassDef) and n.name==name)
        b=next(n for n in copied.body if isinstance(n,ast.ClassDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b),name
    stats['original_ViT_stem_blocks_FFN_unchanged']=True
    old_conv=parsed(root/'mldc_vitmatte_neck_cnl/model/vitmatte_official.py')
    conv=parsed(package/'model/convstream.py')
    for name in ['Basic_Conv3x3','ConvStream']:
        a=next(n for n in old_conv.body if isinstance(n,ast.ClassDef) and n.name==name)
        b=next(n for n in conv.body if isinstance(n,ast.ClassDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b),name
    stats['ConvStream_classes_match_original_DCM']=True
    source=Path(cfg.TEXTURE.TEACHER_INIT_PATH); source_before=source.stat()
    torch.manual_seed(1234); raw=TextureTeacher(nc,'raw')
    torch.manual_seed(1234); teacher=TextureTeacher(nc,'texture')
    assert all(torch.equal(p,dict(raw.named_parameters())[n]) for n,p in teacher.named_parameters())
    stats['teacher_parameters']=sum(p.numel() for p in teacher.parameters())
    assert stats['teacher_parameters']<1000000
    raw.initialize_convstream(source); teacher.initialize_convstream(source)
    assert all(torch.equal(p,dict(raw.convstream.named_parameters())[n]) for n,p in teacher.convstream.named_parameters())
    with torch.no_grad():
        maps=teacher.convstream.eval()(images)
        assert [list(maps['D'+str(i)].shape[1:]) for i in [1,2,3]]==[[48,128,64],[96,64,32],[192,32,16]]
    stats['pretrained_DCM_stream_loaded_strictly']=True
    del raw,maps
    teacher.train(); buffers={n:b.clone() for n,b in teacher.convstream.named_buffers()}
    first=teacher(images); second=teacher(exposure_view(images,.7))
    illumination,structure=specialization_consistency(first,second)
    identity=F.cross_entropy(first['logits'],labels)+TripletLoss(.3)(first['raw'],labels)[0]
    (identity+.1*illumination+.1*structure).backward()
    assert teacher.convstream.convs[0].conv.weight.grad.abs().sum()>0
    assert teacher.encoder[0].weight.grad.abs().sum()>0 and teacher.classifier.weight.grad.abs().sum()>0
    assert all(torch.equal(b,buffers[n]) for n,b in teacher.convstream.named_buffers())
    stats['teacher_identity_texture_illumination_backward']=True
    stats['teacher_specialization_BN_statistics_unchanged']=True
    teacher.zero_grad(set_to_none=True); teacher.eval()
    del first,second,illumination,structure,identity; gc.collect()
    # A deterministic local matching case: shared ID has identical part descriptors.
    tokens=torch.zeros(4,32,128)
    for i in range(4):
        for p in range(32): tokens[i,p,p+(i//2)*32]=1
    scores,confidence=local_pair_scores(tokens,torch.ones(4,32),(8,4))
    assert scores[0,1]==1 and scores[0,2]==0 and confidence[0,1]==1
    zero_scores,zero_conf=local_pair_scores(tokens,torch.zeros(4,32),(8,4))
    assert torch.equal(zero_scores,torch.zeros_like(scores)) and torch.equal(zero_conf,torch.zeros_like(confidence))
    unit=F.normalize(torch.randn(4,768),dim=1).requires_grad_()
    synthetic_labels=torch.tensor([0,0,1,1]); synthetic_cameras=torch.tensor([0,1,0,1])
    loss,details=relative_distillation(unit,scores,confidence,synthetic_labels,synthetic_cameras,indices)
    loss.backward(); assert unit.grad.abs().sum()>0 and details['reliable_triplets']==4
    for cameras_case,indices_case in [(torch.zeros(4,dtype=torch.long),indices),(synthetic_cameras,torch.tensor([0,0,2,2]))]:
        zero,details=relative_distillation(unit,scores,confidence,synthetic_labels,cameras_case,indices_case)
        assert zero==0 and details['reliable_triplets']==0
    bad=torch.ones_like(scores)-scores
    zero,details=relative_distillation(unit,bad,confidence,synthetic_labels,synthetic_cameras,indices)
    assert zero==0 and details['reliable_triplets']==0
    stats['local_matching_and_final_relation_gradient']=True
    stats['same_camera_duplicate_and_unreliable_targets_excluded']=True
    with tempfile.TemporaryDirectory(prefix='convstream_teacher_check_') as temp:
        bad_path=Path(temp)/'untrained.pth'
        torch.save(dict(state_dict=teacher.state_dict(),teacher_metadata={'trained':False}),bad_path)
        try: teacher.load_trained(bad_path)
        except ValueError: stats['untrained_teacher_rejected']=True
        else: raise AssertionError('Untrained teacher accepted')
    torch.manual_seed(1234); baseline=Baseline(cfg,nc,cameras,views).eval(); expected_rng=torch.get_rng_state()
    with torch.inference_mode(): expected,expected_raw,expected_layers=baseline(images[:2],return_components=True)
    baseline_path=Path('/root/log/transreid/nightreid/mldc_learnable_global_cnl/deep_four_uniform_cross_camera_cnl_fixed_equal_schedule/transformer_best.pth')
    before=baseline_path.stat(); baseline.load_param(baseline_path)
    with torch.inference_mode(): trained_expected=baseline(images[:2])
    del baseline; gc.collect()
    torch.manual_seed(1234); model=MLDCReID(cfg,nc,cameras,views,teacher_override=teacher).eval()
    assert torch.equal(torch.get_rng_state(),expected_rng)
    calls=[]; hook=teacher.convstream.register_forward_hook(lambda *args:calls.append(1))
    with torch.inference_mode(): actual,actual_raw,layers=model(images[:2],return_components=True)
    assert torch.equal(actual,expected) and torch.equal(actual_raw,expected_raw) and torch.equal(layers,expected_layers)
    assert not calls and not hasattr(model,'texture_adapter')
    assert all(not block._forward_hooks for block in model.base.blocks)
    stats['SSL_all_CLS_and_final_descriptor_exact']=True; stats['initial_max_error']=float((actual-expected).abs().max())
    stats['no_patch_hooks_or_student_adapter']=True
    checkpoint=torch.load(baseline_path,map_location='cpu',mmap=True,weights_only=False)
    incompatible=model.load_state_dict(checkpoint['state_dict'],strict=False)
    assert not incompatible.unexpected_keys and all(k.startswith('texture_teacher.') for k in incompatible.missing_keys)
    del checkpoint; gc.collect()
    with torch.inference_mode(): actual=model(images[:2])
    assert torch.equal(actual,trained_expected) and not calls
    stats['trained_baseline_final_descriptor_exact']=True
    assert all(p.requires_grad for block in model.base.blocks for p in block.parameters())
    assert all(p.requires_grad for p in model.base.patch_embed.parameters())
    stats['default_entire_active_backbone_trainable']=True
    # CPU memory scope only; the production constructor above does not freeze these blocks.
    model.base.requires_grad_(False); model.base.blocks[-1].requires_grad_(True); model.train()
    teacher_before={n:b.clone() for n,b in teacher.named_buffers()}
    result=model(images,labels,cam_label=camera,view_label=view,return_cnl_features=True,
        return_distillation=True,sample_indices=indices)
    assert len(calls)==2 and torch.isfinite(result[3]['relation'])
    stats['actual_warmstart_targets_on_real_images']=result[3]['stats']
    del result; gc.collect()
    # Controlled targets isolate whether KD reaches the *final* retrieval path.
    original_targets=model.teacher_targets
    synthetic_score=(labels[:,None]==labels[None,:]).float()*.8
    model.teacher_targets=lambda images:(synthetic_score,torch.ones_like(synthetic_score))
    scores_out,features,unit,terms=model(images,labels,cam_label=camera,view_label=view,
        return_cnl_features=True,return_distillation=True,sample_indices=indices)
    unit.retain_grad(); kd=model.distillation_loss(terms,20); assert kd>0
    kd.backward(retain_graph=True)
    assert unit.grad.abs().sum()>0 and model.bottleneck.weight.grad.abs().sum()>0
    assert model.mldc.layer_weights.grad.abs().sum()>0 and model.base.blocks[-1].attn.qkv.weight.grad.abs().sum()>0
    stats['KD_gradient_final_768D_descriptor']=float(unit.grad.norm())
    stats['KD_reaches_BNneck_MLDC_and_final_ViT_block']=True
    cnl=GlobalCNL(labels.tolist(),camera.tolist(),dim=768,gamma=.9,temperature=.05,momentum=.2)
    cnl.update(unit.detach(),indices,labels,camera); cnl_loss,_=cnl(unit,indices,labels,camera)
    combined=F.cross_entropy(scores_out,labels)+TripletLoss(.3)(features,labels)[0]+.1*cnl_loss+kd
    model.zero_grad(set_to_none=True); combined.backward()
    assert torch.isfinite(model.base.blocks[-1].attn.qkv.weight.grad).all()
    assert all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(b,teacher_before[n]) for n,b in teacher.named_buffers())
    stats['combined_ID_Triplet_CNL_final_KD_backward']=True; stats['teacher_frozen_during_student_backward']=True
    model.zero_grad(set_to_none=True); model.teacher_targets=original_targets
    del unit,kd,features,terms,scores_out,combined,cnl,cnl_loss; gc.collect()
    from solver import make_optimizer
    from .loss.make_loss_mldc import make_loss
    loss_fn,center=make_loss(cfg,nc); optimizer,_=make_optimizer(cfg,model,center)
    configure_head_learning_rates(cfg,model,optimizer)
    covered={id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(id(p) not in covered for p in teacher.parameters())
    stats['optimizer_excludes_entire_teacher']=True
    del optimizer,loss_fn,center
    model.texture_enabled=False; count=len(calls)
    with torch.no_grad():
        *_,terms=model(images,labels,cam_label=camera,view_label=view,return_cnl_features=True,
            return_distillation=True,sample_indices=indices)
    assert len(calls)==count and terms['relation']==0
    model.texture_enabled=True; hook.remove(); model.eval()
    stats['disabled_KD_does_not_execute_teacher']=True
    with torch.inference_mode(): expected_export=model(images[:2])
    with tempfile.TemporaryDirectory(prefix='convstream_student_check_') as temp:
        full=Path(temp)/'full.pth'; core=Path(temp)/'core.pth'
        torch.save(dict(state_dict=model.state_dict(),architecture=model.architecture(),config=cfg.dump(),epoch=0,
            mldc_layers=[9,10,11,12],temperature=.9,train_feature='mldc',readout='mldc',
            teacher_metadata={'trained':False,'verification_fixture':True}),full)
        export(full,core); del model,teacher,original_targets; gc.collect()
        restored=MLDCReID(cfg,nc,cameras,views,initialize=False).eval(); restored.load_param(full)
        with torch.inference_mode(): actual=restored(images[:2])
        assert torch.equal(actual,expected_export)
        del restored; gc.collect()
        original=Baseline(cfg,nc,cameras,views,initialize=False).eval(); original.load_param(core)
        with torch.inference_mode(): actual=original(images[:2])
        assert torch.equal(actual,expected_export)
        stats['full_checkpoint_roundtrip_and_original_model_export_exact']=True
        del original; gc.collect()
    after=baseline_path.stat(); source_after=source.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    assert (source_before.st_size,source_before.st_mtime_ns)==(source_after.st_size,source_after.st_mtime_ns)
    stats.update(all_passed=True,seed=1234,dataset='nightreid',real_images=4,device='cpu',optimizer_steps=0,
        teacher_specialization_started=False,student_training_started=False,performance_measured=False,
        CUDA_capacity_tested=False,original_checkpoints_unchanged=True,
        CPU_backward_scope='Only final ViT block and original heads; full backbone trainable by production default',
        nonzero_KD_test_targets='Controlled verification-only pair scores; real warmstart coverage reported separately')
    output=root/'analysis_convstream_final_texture_verification_20261026'; output.mkdir(exist_ok=True)
    (output/'verification.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print('VERIFICATION '+json.dumps(stats),flush=True)


if __name__=='__main__': main()
