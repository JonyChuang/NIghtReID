"""Real-image checks for four direct CLS refiners and one identity prototype."""
import argparse,json,re,tempfile
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
import yaml
from yacs.config import CfgNode
from .model.make_model_mldc import MLDCReID,add_mldc_config
from .model.base_model import MLDCReID as Baseline
from .model.prototype_attention import attention_distillation
from .prototype_memory import PrototypeMemory
from .global_cnl import GlobalCNL
from .training_updates import commit_training_memories
from .loss.make_loss_mldc import make_loss


def unchanged(state,module):
    return all(torch.equal(v,module.state_dict()[k]) for k,v in state.items())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',default='/root/log/transreid/nightreid/mldc_learnable_global_cnl/deep_four_uniform_cross_camera_cnl_fixed_equal_schedule/transformer_best.pth')
    parser.add_argument('--train-dir',default='data/NightReID/bounding_box_train')
    parser.add_argument('--report',default=str(Path(__file__).parent/'verification_multilevel_cls.json'))
    args=parser.parse_args();torch.set_num_threads(4);torch.manual_seed(1234)
    cp=torch.load(args.baseline,map_location='cpu',mmap=True,weights_only=False)
    cfg=add_mldc_config(CfgNode(yaml.safe_load(cp['config'])))
    cfg.MODEL.PRETRAIN_CHOICE='self';cfg.MODEL.PRETRAIN_PATH=args.baseline
    del cp
    groups=defaultdict(dict);cmap={'L1':0,'L2':1,'L3':2,'R1':3,'R2':4,'R3':5}
    for p in sorted(Path(args.train_dir).glob('*.jpg')):
        m=re.match(r'(\d{4})([LR][123])C',p.name)
        if m and 1<=int(m[1])<=500:groups[int(m[1])].setdefault(m[2],p)
    chosen=[(y,sorted(c.items())[:2]) for y,c in sorted(groups.items()) if len(c)>=2][:2]
    assert len(chosen)==2
    images=[];labels=[];cams=[];names=[]
    for y,views in chosen:
        for cam,p in views:
            with Image.open(p) as im:
                a=np.array(im.convert('RGB').resize((128,256),Image.BILINEAR),dtype=np.float32)/255
            images.append((torch.from_numpy(a).permute(2,0,1)-.5)/.5)
            labels.append(y-1);cams.append(cmap[cam]);names.append(p.name)
    images=torch.stack(images);labels=torch.tensor(labels);cams=torch.tensor(cams);indices=torch.arange(4)
    model=MLDCReID(cfg,500,6,1,initialize=False).eval();model.load_baseline(args.baseline)
    levels=model.feature_layers
    assert levels==(9,10,11,12)
    assert len(model.pad_head.level_heads)==4
    assert not any('ffn' in n.lower() for n,_ in model.pad_head.named_parameters())
    assert all(not isinstance(m,torch.nn.BatchNorm1d) for m in model.pad_head.modules())
    all_ids=[{id(p) for p in h.parameters()} for h in model.pad_head.level_heads.values()]
    assert all(not (a&b) for i,a in enumerate(all_ids) for b in all_ids[i+1:])
    backbone_before={n:p.clone() for n,p in model.base.state_dict().items()}
    captures={};hooks=[]
    for level in levels:
        def capture(m,a,o,l=level):captures[l]=(o[:,0].detach().clone(),o[:,1:].detach().clone())
        hooks.append(model.base.blocks[level-1].register_forward_hook(capture))
    with torch.no_grad():
        reference=Baseline.forward(model,images)
        initial=model(images,return_pad_details=True)
    for h in hooks:h.remove()
    for level in levels:
        assert torch.equal(initial['patches'][level],captures[level][1])
        assert torch.equal(initial['layers'][level-1],captures[level][0])
    initial_error=float((reference-initial['retrieval']).abs().max());assert initial_error==0
    assert initial['student']['attention'].shape==(4,4,8,128)
    assert torch.allclose(initial['student']['attention'].sum(-1),torch.ones(4,4,8),atol=1e-6)
    assert initial['student']['unit'] is initial['student_features']
    assert initial['student_features'].shape==(4,768)
    with torch.no_grad():
        legacy_cls,legacy_final=model.base(images,return_final_patches=True)
        raw_cls,selected=model.base(images,return_patch_layers=levels)
    assert torch.equal(legacy_cls,raw_cls) and torch.equal(legacy_final,selected[12])
    model.pad_enabled=False
    with torch.no_grad():assert torch.equal(reference,model(images))
    model.pad_enabled=True
    for head in model.pad_head.level_heads.values():
        with torch.no_grad():head.output_projection.weight.normal_(0,.01)
    raw=initial['layers'].detach()
    patches={l:p.detach() for l,p in initial['patches'].items()}
    with torch.no_grad():
        base=model.pad_head.student(raw,patches)
        changed=dict(patches);changed[9]=patches[9].flip(-1)
        perturbed=model.pad_head.student(raw,changed)
    assert not torch.equal(base['levels'][9]['attention'],perturbed['levels'][9]['attention'])
    assert not torch.equal(base['levels'][9]['refined_cls'],perturbed['levels'][9]['refined_cls'])
    for level in [10,11,12]:
        assert torch.equal(base['levels'][level]['attention'],perturbed['levels'][level]['attention'])
        assert torch.equal(base['levels'][level]['refined_cls'],perturbed['levels'][level]['refined_cls'])
    head=model.pad_head.level_heads['9'];sp=base['spatial'][9]
    q=head.query_projection(head.query_norm(raw[8])).reshape(4,8,1,head.head_dim)
    expected_log=(q.float()@sp['key'].float().transpose(-1,-2)/head.head_dim**.5).log_softmax(-1)
    assert torch.allclose(base['levels'][9]['log_attention'],expected_log[:,:,0],atol=1e-6)
    context=(expected_log.exp()@sp['value']).transpose(1,2).reshape(4,head.dim)
    assert torch.allclose(base['levels'][9]['context'],context,atol=1e-6)
    assert torch.allclose(base['levels'][9]['refined_cls'],raw[8]+head.output_projection(context),atol=1e-6)
    model.train();model.base.eval().requires_grad_(False)
    bank=PrototypeMemory(labels.tolist(),500,768,update_mode='adaptive',amu_temperature=.05)
    cnl=GlobalCNL(labels.tolist(),cams.tolist(),dim=768)
    snap={k:v.clone() for k,v in bank.state_dict().items()}
    cnl_before={k:v.clone() for k,v in cnl.state_dict().items()}
    warm=model(images,labels,cam_label=cams,return_cnl_features=True,return_distillation=True,
               sample_indices=indices,prototype_memory=bank,epoch=1)
    assert warm[2] is warm[3]['student_features'] and warm[3]['stats']['teacher_anchors']==0
    assert float(warm[3]['attention'])==0 and bank.features.shape==(500,768)
    assert unchanged(snap,bank) and unchanged(cnl_before,cnl)
    commit_training_memories(False,cnl,bank,warm[2],warm[3],indices,labels,cams)
    assert unchanged(snap,bank) and unchanged(cnl_before,cnl)
    commit_training_memories(True,cnl,bank,warm[2],warm[3],indices,labels,cams)
    assert bank.image_counts[labels.unique()].tolist()==[2,2]
    assert bank.updates[labels.unique()].tolist()==[2,2]
    assert torch.allclose(cnl.features,warm[2].detach())
    counts={l:0 for l in levels};calls={l:[] for l in levels};hooks=[]
    for level in levels:
        def called(m,a,o,l=level):
            counts[l]+=1;calls[l].append((a[0],a[1],a[2]))
        hooks.append(model.pad_head.level_heads[str(level)].register_forward_hook(called))
    bn=[];hooks.append(model.bottleneck.register_forward_hook(lambda m,a,o:bn.append(o)))
    n=int(model.bottleneck.num_batches_tracked);snap={k:v.clone() for k,v in bank.state_dict().items()}
    # Frozen real-image backbone: mark its actual output nodes as differentiable
    # leaves to verify the route back to each CLS/patch without a full ViT backward.
    def mark_feature_nodes(module,args,output):
        raw,patch=output
        return raw.requires_grad_(),{l:value.requires_grad_() for l,value in patch.items()}
    grad_hook=model.base.register_forward_hook(mark_feature_nodes)
    d=model(images,labels,cam_label=cams,return_pad_details=True,
            sample_indices=indices,prototype_memory=bank,epoch=6)
    grad_hook.remove()
    for h in hooks:h.remove()
    assert all(x==2 for x in counts.values())
    same_prototype=calls[9][1][0]
    for level in levels:
        assert calls[level][1][0] is same_prototype
        assert torch.equal(calls[level][0][0],d['layers'][level-1])
        assert torch.equal(calls[level][1][0],bank.features[labels])
    assert int(model.bottleneck.num_batches_tracked)==n+2
    assert d['student_features'] is d['student']['unit'] is d['terms']['student_features']
    assert torch.equal(d['student_features'],F.normalize(bn[0].float(),dim=1))
    assert d['teacher']['unit'].shape==(4,768)
    weights=model.readout_weights()
    expected=sum(d['refined_layers'][level-1]*weights[level-1] for level in levels)
    assert torch.allclose(d['mixed'],expected,atol=1e-6)
    teacher_expected=sum(d['teacher']['refined_layers'][level-1]*weights[level-1] for level in levels)
    assert torch.allclose(d['teacher']['mixed'],teacher_expected,atol=1e-6)
    keys,columns=bank.classification_keys(d['student_features'],labels,indices)
    expected_id=F.cross_entropy(d['student_features']@keys.T/cfg.PAD.PROTO_TEMPERATURE,columns)
    assert torch.allclose(d['terms']['student_id'],expected_id)
    per=F.kl_div(d['student']['log_attention'],d['teacher']['attention'].detach(),reduction='none').sum(-1)
    assert torch.allclose(d['terms']['attention'],per.mean(),atol=1e-7)
    for i,level in enumerate(levels):
        assert abs(d['terms']['stats']['attention_kl_layer%d'%level]-float(per[:,i].mean()))<1e-7
    assert torch.allclose(d['terms']['update_features'],d['teacher']['unit'].detach())
    assert unchanged(snap,bank)
    model.zero_grad(set_to_none=True)
    d['teacher']['attention'].retain_grad()
    kl=d['terms']['attention']
    grad_cls=torch.autograd.grad(kl,d['layers'],retain_graph=True)[0]
    grad_patches=torch.autograd.grad(kl,tuple(d['patches'][l] for l in levels),retain_graph=True)
    cls_norms={str(l):float(grad_cls[l-1].double().norm()) for l in levels}
    patch_norms={str(l):float(g.double().norm()) for l,g in zip(levels,grad_patches)}
    assert all(x>0 for x in cls_norms.values()) and all(x>0 for x in patch_norms.values())
    assert torch.count_nonzero(grad_cls[:8])==0
    kl.backward(retain_graph=True)
    assert d['teacher']['attention'].grad is None
    for h in model.pad_head.level_heads.values():
        assert h.query_projection.weight.grad is not None and h.query_projection.weight.grad.double().norm()>0
        assert h.output_projection.weight.grad is None
    model.zero_grad(set_to_none=True)
    loss_fn,_=make_loss(cfg,500)
    scores=model.classifier(model.dropout(d['retrieval']))
    original=loss_fn(scores,d['mixed'],labels,cams)
    cnl_loss,_=cnl(d['student_features'],indices,labels,cams)
    total=original+.1*cnl_loss+model.distillation_loss(d['terms'],6)
    total.backward()
    output_gradients={str(l):float(model.pad_head.level_heads[str(l)].output_projection.weight.grad.double().norm()) for l in levels}
    assert all(x>0 for x in output_gradients.values())
    assert model.mldc.layer_weights.grad is not None and model.mldc.layer_weights.grad.double().norm()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.pad_head.parameters())
    update_before=bank.updates.clone()
    commit_training_memories(False,cnl,bank,d['student_features'],d['terms'],indices,labels,cams)
    assert unchanged(snap,bank)
    commit_training_memories(True,cnl,bank,d['student_features'],d['terms'],indices,labels,cams)
    assert (bank.updates-update_before)[labels.unique()].tolist()==[2,2]
    assert bank.image_counts[labels.unique()].tolist()==[2,2]
    model.eval();counts={l:0 for l in levels};hooks=[]
    for level in levels:
        def counted(m,a,o,l=level):counts[l]+=1
        hooks.append(model.pad_head.level_heads[str(level)].register_forward_hook(counted))
    n=int(model.bottleneck.num_batches_tracked)
    with torch.no_grad():
        expected=model(images[:2]).clone();single=model(images[:1])
    for h in hooks:h.remove()
    assert all(x==2 for x in counts.values()) and int(model.bottleneck.num_batches_tracked)==n
    assert torch.isfinite(single).all() and not torch.equal(expected,reference[:2])
    with torch.no_grad():parts=model(images[:2],return_pad_details=True)
    assert torch.equal(parts['retrieval'],expected)
    try:model(images[:2],prototype_memory=bank)
    except ValueError:pass
    else:raise AssertionError('Inference accessed training identity prototypes')
    assert all(torch.equal(v,backbone_before[n]) for n,v in model.base.state_dict().items())
    with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
        s=model.pad_head.student(raw,patches)
        t=model.pad_head.teacher(bank.features[labels],raw,s['spatial'])
        low=attention_distillation(s['log_attention'],t['attention'])
    assert torch.isfinite(low) and s['attention'].dtype==torch.float32
    from .processor.processor_mldc import save_checkpoint
    from .export_inference import export
    with tempfile.TemporaryDirectory(prefix='multilevel_cls_verify_',dir=str(Path(args.report).resolve().parent)) as directory:
        full=Path(directory)/'full.pth';inference=Path(directory)/'inference.pth'
        save_checkpoint(model,cfg,0,full,memory=cnl,prototype_memory=bank)
        model.load_param(str(full))
        with torch.no_grad():assert torch.equal(model(images[:2]),expected)
        export(full,inference);model.load_param(str(inference))
        with torch.no_grad():assert torch.equal(model(images[:2]),expected)
        restore=PrototypeMemory(labels.tolist(),500,768);restore.load_state_dict(bank.state_dict())
        assert unchanged(bank.state_dict(),restore)
    from .verify_paper_alignment import check_paper_alignment
    paper=check_paper_alignment(model)
    result=dict(status='passed',optimizer_steps=0,training_launched=False,real_training_images=names,
        feature_layers=list(levels),patches_match_actual_block_outputs=True,attention_shape=[4,4,8,128],
        direct_attention_manual_reference=True,no_ffn_or_decoder_stack=True,
        per_level_parameters_independent=True,teacher_student_share_per_level_projections=True,
        same_single_prototype_input_all_four_levels=True,prototype_shape=[500,768],
        patch9_perturbation_isolated_to_level9=True,initial_baseline_max_abs_error=initial_error,
        refined_cls_aggregated_by_original_mldc=True,teacher_uses_same_mldc_weights=True,
        kl_mean_over_batch_four_levels_and_heads=True,cls_kl_gradient_norms=cls_norms,
        patch_kl_gradient_norms=patch_norms,unselected_cls_have_no_direct_kl_gradient=True,
        teacher_kl_target_detached=True,identity_loss_updates_all_four_refiners=True,
        output_projection_gradient_norms=output_gradients,original_id_triplet_factory_verified=True,
        cnl_and_student_prototype_ce_share_final_descriptor=True,
        amu_updated_once_per_image_not_once_per_level=True,forward_does_not_write_memory=True,
        amp_skip_does_not_write_memories=True,shared_final_bn_for_student_and_teacher=True,
        inference_retains_all_four_refiners=True,inference_has_no_prototype_access=True,
        checkpoint_roundtrip_exact=True,inference_export_exact=True,low_precision_finite=True,
        backbone_weights_and_buffers_unchanged=True,paper_alignment=paper,
        verification_note='Four real training images; frozen backbone with differentiable encoder-output leaves and nonzero refinement fixture for gradient checks. No optimizer steps or new retrieval metrics.')
    Path(args.report).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
