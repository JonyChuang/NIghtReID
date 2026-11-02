"""Verify one fused student, real-image losses, KL routing and inference."""
import argparse,json,re,tempfile
from pathlib import Path
from collections import defaultdict
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import yaml
from yacs.config import CfgNode
from .model.make_model_mldc import MLDCReID,add_mldc_config
from .model.base_model import MLDCReID as Baseline
from .model.prototype_attention import attention_distillation
from .prototype_memory import PrototypeMemory
from .global_cnl import GlobalCNL
from .training_updates import commit_training_memories


def unchanged(state,module):
    return all(torch.equal(v,module.state_dict()[k]) for k,v in state.items())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',default='/root/log/transreid/nightreid/mldc_learnable_global_cnl/deep_four_uniform_cross_camera_cnl_fixed_equal_schedule/transformer_best.pth')
    parser.add_argument('--train-dir',default='data/NightReID/bounding_box_train')
    parser.add_argument('--report',default=str(Path(__file__).parent/'verification_fused_student.json'))
    args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(1234)
    checkpoint=torch.load(args.baseline,map_location='cpu',mmap=True,weights_only=False)
    cfg=add_mldc_config(CfgNode(yaml.safe_load(checkpoint['config'])))
    cfg.MODEL.PRETRAIN_CHOICE='self';cfg.MODEL.PRETRAIN_PATH=args.baseline
    del checkpoint
    groups=defaultdict(dict);camdict={'L1':0,'L2':1,'L3':2,'R1':3,'R2':4,'R3':5}
    for p in sorted(Path(args.train_dir).glob('*.jpg')):
        m=re.match(r'(\d{4})([LR][123])C',p.name)
        if m and 1<=int(m[1])<=500:groups[int(m[1])].setdefault(m[2],p)
    chosen=[(pid,sorted(cams.items())[:2]) for pid,cams in sorted(groups.items()) if len(cams)>=2][:2]
    assert len(chosen)==2,'Need two real identities with cross-camera images'
    images=[];labels=[];cameras=[];filenames=[]
    for pid,views in chosen:
        for cam,p in views:
            with Image.open(p) as im:a=np.array(im.convert('RGB').resize((128,256),Image.BILINEAR),dtype=np.float32)/255
            images.append((torch.from_numpy(a).permute(2,0,1)-.5)/.5)
            labels.append(pid-1);cameras.append(camdict[cam]);filenames.append(p.name)
    images=torch.stack(images);labels=torch.tensor(labels);cameras=torch.tensor(cameras);indices=torch.arange(4)
    model=MLDCReID(cfg,500,6,1,initialize=False).eval();model.load_baseline(args.baseline)
    assert not hasattr(model.pad_head,'embedding')
    assert not any(isinstance(m,torch.nn.BatchNorm1d) for m in model.pad_head.modules())
    backbone_before={n:p.clone() for n,p in model.base.state_dict().items()}
    with torch.no_grad():
        reference=Baseline.forward(model,images)
        initial=model(images,return_pad_details=True)
        initial_error=float((reference-initial['retrieval']).abs().max())
    assert initial_error==0
    assert initial['student']['unit'] is initial['student_features']
    assert initial['student_features'].shape==(4,768)
    assert torch.equal(initial['student_features'],F.normalize(initial['retrieval'].float(),dim=1))
    assert initial['student']['attention'].shape==(4,3,8,128)
    assert torch.allclose(initial['student']['attention'].sum(-1),torch.ones(4,3,8),atol=1e-6)
    captured=[];hook=model.base.blocks[-1].register_forward_hook(lambda m,a,o:captured.append(o[:,1:].detach()))
    with torch.no_grad():probe=model(images,return_pad_details=True)
    hook.remove();assert torch.equal(probe['patches'],captured[0])
    model.pad_enabled=False
    with torch.no_grad():assert torch.equal(reference,model(images))
    model.pad_enabled=True;model.train();model.base.eval().requires_grad_(False)
    bank=PrototypeMemory(labels.tolist(),500,768,update_mode=cfg.PAD.UPDATE_MODE,amu_temperature=cfg.PAD.AMU_TEMPERATURE)
    cnl=GlobalCNL(labels.tolist(),cameras.tolist(),dim=768)
    before={k:v.clone() for k,v in bank.state_dict().items()}
    cnl_before={k:v.clone() for k,v in cnl.state_dict().items()}
    n=int(model.bottleneck.num_batches_tracked)
    warm=model(images,labels,cam_label=cameras,return_cnl_features=True,return_distillation=True,
        sample_indices=indices,prototype_memory=bank,epoch=1)
    assert warm[2] is warm[3]['student_features']
    assert warm[2].shape==(4,768) and bank.features.shape==(500,768)
    assert warm[3]['stats']['teacher_anchors']==0 and float(warm[3]['attention'])==0
    assert int(model.bottleneck.num_batches_tracked)==n+1
    assert unchanged(before,bank) and unchanged(cnl_before,cnl)
    keys,columns=bank.classification_keys(warm[2],labels,indices)
    assert torch.allclose(warm[3]['student_id'],F.cross_entropy(warm[2]@keys.T/cfg.PAD.PROTO_TEMPERATURE,columns))
    commit_training_memories(False,cnl,bank,warm[2],warm[3],indices,labels,cameras)
    assert unchanged(before,bank) and unchanged(cnl_before,cnl)
    commit_training_memories(True,cnl,bank,warm[2],warm[3],indices,labels,cameras)
    assert bank.image_counts[labels.unique()].tolist()==[2,2]
    assert torch.allclose(cnl.features,warm[2].detach(),atol=1e-6)
    # Context projections are zero-initialized. After warmup a nonzero projection
    # is required to test gradients through fusion and teacher contexts.
    with torch.no_grad():model.pad_head.residual_projection.weight.normal_(0,.01)
    counts=[0,0,0];hooks=[]
    for i,l in enumerate(model.pad_head.layers):
        def count(m,a,o,index=i):counts[index]+=1
        hooks.append(l.register_forward_hook(count))
    query_inputs=[];BN_outputs=[]
    hooks.append(model.pad_head.student_query.register_forward_pre_hook(lambda m,a:query_inputs.append(a[0])))
    hooks.append(model.bottleneck.register_forward_hook(lambda m,a,o:BN_outputs.append(o)))
    n=int(model.bottleneck.num_batches_tracked);snap={k:v.clone() for k,v in bank.state_dict().items()}
    scores,mixed,unit,terms=model(images,labels,cam_label=cameras,return_cnl_features=True,
        return_distillation=True,sample_indices=indices,prototype_memory=bank,epoch=6)
    for h in hooks:h.remove()
    assert counts==[3,3,3]  # fusion context, final-student attention, prototype teacher
    assert len(query_inputs)==1 and query_inputs[0] is unit
    assert unit is terms['student_features']
    assert terms['teacher_features'].shape==(4,768)
    assert torch.equal(unit,F.normalize(BN_outputs[0].float(),dim=1))
    assert int(model.bottleneck.num_batches_tracked)==n+2  # one common BN, two identity paths
    assert unchanged(snap,bank)
    keys,columns=bank.classification_keys(unit,labels,indices)
    assert torch.allclose(terms['student_id'],F.cross_entropy(unit@keys.T/cfg.PAD.PROTO_TEMPERATURE,columns))
    assert torch.allclose(terms['update_features'],terms['teacher_features'].detach())
    cnl_loss,_=cnl(unit,indices,labels,cameras)
    total=F.cross_entropy(scores.float(),labels)+.1*cnl_loss+model.distillation_loss(terms,6)
    total.backward()
    prefixes=['spatial_projection','fusion_query','student_query','prototype_query','layers.0','layers.1','layers.2','residual_projection']
    gradients={p:sum(float(v.grad.float().square().sum()) for name,v in model.pad_head.named_parameters() if name.startswith(p) and v.grad is not None)**.5 for p in prefixes}
    assert all(v>0 and np.isfinite(v) for v in gradients.values()),gradients
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.pad_head.parameters())
    model.zero_grad(set_to_none=True)
    details=model(images,labels,cam_label=cameras,return_pad_details=True,
        sample_indices=indices,prototype_memory=bank,epoch=6)
    teacher=details['teacher'];student=details['student']
    assert student['unit'] is details['student_features'] is details['terms']['student_features']
    teacher['attention'].retain_grad()
    details['mixed'].retain_grad();student['unit'].retain_grad()
    kl=details['terms']['attention']
    # These are gradients at the actual final descriptor/fusion nodes. Renaming
    # the old auxiliary output or detaching the fusion would fail this check.
    fused_grad,unit_grad=torch.autograd.grad(kl,(details['mixed'],student['unit']),retain_graph=True)
    assert fused_grad.norm()>0 and unit_grad.norm()>0
    kl.backward(retain_graph=True)
    assert teacher['attention'].grad is None
    assert model.pad_head.prototype_query[1].weight.grad is None
    assert model.pad_head.student_query[1].weight.grad.norm()>0
    assert model.pad_head.layers[0].query.weight.grad.norm()>0
    assert model.pad_head.residual_projection.weight.grad.norm()>0
    model.zero_grad(set_to_none=True)
    details['terms']['teacher_id'].backward()
    assert model.pad_head.prototype_query[1].weight.grad.norm()>0
    teacher_id_gradient=float(model.pad_head.prototype_query[1].weight.grad.norm())
    # Warmup and active teacher writes occur only at explicit commit points.
    model.zero_grad(set_to_none=True)
    assert unchanged(snap,bank)
    before_update=bank.features.clone()
    commit_training_memories(False,cnl,bank,details['student_features'],details['terms'],indices,labels,cameras)
    assert torch.equal(before_update,bank.features)
    commit_training_memories(True,cnl,bank,details['student_features'],details['terms'],indices,labels,cameras)
    assert bank.features.shape==(500,768) and torch.isfinite(bank.features).all()
    model.eval();n=int(model.bottleneck.num_batches_tracked)
    counts=[0,0,0];hooks=[]
    for i,l in enumerate(model.pad_head.layers):
        def count(m,a,o,index=i):counts[index]+=1
        hooks.append(l.register_forward_hook(count))
    with torch.no_grad():expected=model(images[:2]).clone();single=model(images[:1])
    for h in hooks:h.remove()
    assert counts==[2,2,2]  # two normal inference calls, fusion path only
    assert torch.isfinite(single).all() and int(model.bottleneck.num_batches_tracked)==n
    with torch.no_grad():expanded=model(images[:2],return_pad_details=True)
    assert torch.equal(expected,expanded['retrieval'])
    assert not torch.equal(expected,reference[:2])
    try:model(images[:2],prototype_memory=bank)
    except ValueError:pass
    else:raise AssertionError('Inference accessed identity prototypes')
    assert all(torch.equal(p,backbone_before[n]) for n,p in model.base.state_dict().items())
    with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
        sp=model.pad_head.prepare(initial['patches'])
        ss=model.pad_head.student(initial['student_features'],sp)
        tt=model.pad_head.teacher(bank.features[labels],sp)
        low_precision=attention_distillation(ss['log_attention'],tt['attention'])
    assert torch.isfinite(low_precision) and ss['attention'].dtype==torch.float32
    from .processor.processor_mldc import save_checkpoint
    from .export_inference import export
    with tempfile.TemporaryDirectory(prefix='fused_pad_check_',dir=str(Path(args.report).resolve().parent)) as directory:
        cp=Path(directory)/'full.pth';ep=Path(directory)/'inference.pth'
        save_checkpoint(model,cfg,0,cp,memory=cnl,prototype_memory=bank)
        model.load_param(str(cp))
        with torch.no_grad():assert torch.equal(expected,model(images[:2]))
        export(cp,ep);model.load_param(str(ep))
        with torch.no_grad():assert torch.equal(expected,model(images[:2]))
        restored=PrototypeMemory(labels.tolist(),500,768);restored.load_state_dict(bank.state_dict())
        assert unchanged(bank.state_dict(),restored)
    from .verify_paper_alignment import check_paper_alignment
    paper=check_paper_alignment(model)
    result=dict(status='passed',optimizer_steps=0,training_launched=False,real_training_images=filenames,
        initial_baseline_max_abs_error=initial_error,p12_source_exact=True,
        one_fused_student_identity=True,identity_dimension=768,attention_internal_dimension=256,
        independent_256_identity_embedding_removed=True,student_attention_queries_exact_final_cnl_tensor=True,
        student_prototype_ce_uses_exact_final_descriptor=True,prototype_shape=[500,768],
        shared_layer_calls_training=[3,3,3],normal_inference_only_constructs_fusion=True,
        final_fusion_kl_gradient_norm=float(fused_grad.norm()),final_descriptor_kl_gradient_norm=float(unit_grad.norm()),
        kl_teacher_target_detached=True,kl_student_shared_layers_and_fusion_receive_gradient=True,
        teacher_identity_gradient_norm=teacher_id_gradient,gradient_norms=gradients,
        shared_final_bn_updated_for_student_and_teacher=True,
        forward_does_not_write_memory=True,amp_skip_does_not_write_memories=True,
        teacher_refined_features_feed_amu=True,backbone_weights_and_buffers_unchanged=True,
        inference_rejects_prototypes=True,low_precision_finite=True,
        checkpoint_roundtrip_exact=True,inference_export_retains_fusion=True,paper_alignment=paper,
        verification_note='Real training images, frozen backbone; nonzero context projection fixture for gradient checks. No optimizer steps.')
    Path(args.report).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
