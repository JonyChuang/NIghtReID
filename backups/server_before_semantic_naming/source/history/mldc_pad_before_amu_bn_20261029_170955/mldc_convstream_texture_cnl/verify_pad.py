"""Real-image PAD integration checks. No optimizer steps or training run."""
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
    parser.add_argument('--report',default=str(Path(__file__).parent/'verification_pad.json'))
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
    assert len(chosen)==2,'Need two real training identities with two camera views each'
    images=[];labels=[];cameras=[];filenames=[]
    for pid,views in chosen:
        for cam,p in views:
            with Image.open(p) as im:a=np.array(im.convert('RGB').resize((128,256),Image.BILINEAR),dtype=np.float32)/255
            images.append((torch.from_numpy(a).permute(2,0,1)-.5)/.5)
            labels.append(pid-1);cameras.append(camdict[cam]);filenames.append(p.name)
    images=torch.stack(images);labels=torch.tensor(labels);cameras=torch.tensor(cameras);indices=torch.arange(4)
    model=MLDCReID(cfg,500,6,1,initialize=False).eval();model.load_baseline(args.baseline)
    with torch.no_grad():
        reference=Baseline.forward(model,images)
        details=model(images,return_pad_details=True)
        initial_error=float((reference-details['retrieval']).abs().max())
    assert initial_error==0
    assert details['patches'].shape==(4,128,768)
    assert details['student']['attention'].shape==(4,3,8,128)
    assert torch.allclose(details['student']['attention'].sum(-1),torch.ones(4,3,8),atol=1e-6)
    assert len({id(l.query.weight) for l in model.pad_head.layers})==3
    captured=[];handle=model.base.blocks[-1].register_forward_hook(lambda m,a,o:captured.append(o[:,1:].detach()))
    with torch.no_grad():probe=model(images,return_pad_details=True)
    handle.remove();assert torch.equal(probe['patches'],captured[0])
    model.pad_enabled=False
    with torch.no_grad():assert torch.equal(reference,model(images))
    model.pad_enabled=True;model.train();model.base.eval().requires_grad_(False)
    bank=PrototypeMemory(labels.tolist(),500,cfg.PAD.DIM,cfg.PAD.MOMENTUM,cfg.PAD.MIN_IMAGES)
    cnl=GlobalCNL(labels.tolist(),cameras.tolist(),dim=768)
    bank_before={k:v.clone() for k,v in bank.state_dict().items()};cnl_before={k:v.clone() for k,v in cnl.state_dict().items()}
    bn_before=int(model.bottleneck.num_batches_tracked)
    warm=model(images,labels,cam_label=cameras,return_cnl_features=True,return_distillation=True,
        sample_indices=indices,prototype_memory=bank,epoch=1)
    assert warm[3]['stats']['teacher_anchors']==0 and float(warm[3]['attention'])==0
    assert int(model.bottleneck.num_batches_tracked)==bn_before+1
    assert unchanged(bank_before,bank) and unchanged(cnl_before,cnl)
    commit_training_memories(False,cnl,bank,warm[2],warm[3],indices,labels,cameras)
    assert unchanged(bank_before,bank) and unchanged(cnl_before,cnl)
    commit_training_memories(True,cnl,bank,warm[2],warm[3],indices,labels,cameras)
    assert bank.image_counts[labels.unique()].tolist()==[2,2]
    assert torch.allclose(cnl.features,warm[2].detach(),atol=1e-6)
    duplicate=PrototypeMemory([int(labels[0]),int(labels[2])],500,256,min_images=2)
    dl=torch.stack([labels[0],labels[0],labels[2]]);di=torch.tensor([0,0,1])
    for _ in range(2):duplicate.update(warm[3]['update_features'][:3],dl,di)
    assert duplicate.image_counts[labels.unique()].tolist()==[1,1] and not duplicate.lookup(labels)[1].any()
    counts=[0,0,0];hooks=[]
    for i,l in enumerate(model.pad_head.layers):
        def count(m,a,o,index=i):counts[index]+=1
        hooks.append(l.register_forward_hook(count))
    snap={k:v.clone() for k,v in bank.state_dict().items()};bn_before=int(model.bottleneck.num_batches_tracked)
    bn_values=[];bh=model.bottleneck.register_forward_hook(lambda m,a,o:bn_values.append(o))
    scores,mixed,unit,terms=model(images,labels,cam_label=cameras,return_cnl_features=True,
        return_distillation=True,sample_indices=indices,prototype_memory=bank,epoch=6)
    bh.remove()
    for h in hooks:h.remove()
    assert counts==[2,2,2] and terms['stats']['teacher_anchors']==4
    assert int(model.bottleneck.num_batches_tracked)==bn_before+1
    assert torch.equal(unit,F.normalize(bn_values[0].float(),dim=1)) and unchanged(snap,bank)
    cnl_loss,_=cnl(unit,indices,labels,cameras)
    total=F.cross_entropy(scores.float(),labels)+.1*cnl_loss+model.distillation_loss(terms,6)
    total.backward()
    prefixes=['spatial_projection','student_query','prototype_query','layers.0','layers.1','layers.2','embedding','residual_projection']
    gradient={p:sum(float(v.grad.float().square().sum()) for n,v in model.pad_head.named_parameters() if n.startswith(p) and v.grad is not None)**.5 for p in prefixes}
    assert all(x>0 and np.isfinite(x) for x in gradient.values()),gradient
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.pad_head.parameters())
    model.zero_grad(set_to_none=True)
    g=details['global_feature'].detach().requires_grad_();p=details['patches'].detach().requires_grad_()
    f,t,_=model._pad(g,p,labels,indices,bank,epoch=6)
    gg,pg=torch.autograd.grad(model.distillation_loss(t,6)+f.square().mean(),(g,p))
    assert gg.norm()>0 and pg.norm()>0
    sl=torch.randn(2,3,8,128,requires_grad=True);tl=torch.randn_like(sl,requires_grad=True)
    logs=sl.log_softmax(-1);teacher=tl.softmax(-1);kl=attention_distillation(logs,teacher)
    manual=(teacher.detach()*(teacher.detach().log()-logs)).sum(-1).mean()
    assert torch.allclose(kl,manual,atol=1e-7)
    kl.backward();assert sl.grad is not None and tl.grad is None
    with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
        sp=model.pad_head.prepare(details['patches']);ss=model.pad_head.student(details['global_feature'],sp)
        tt=model.pad_head.teacher(bank.features[labels],sp)
        low_precision=attention_distillation(ss['log_attention'],tt['attention'])
    assert torch.isfinite(low_precision) and ss['attention'].dtype==torch.float32
    with torch.no_grad():model.pad_head.residual_projection.weight.normal_(0,.01)
    model.eval()
    with torch.no_grad():expected=model(images[:2]).clone()
    assert not torch.equal(expected,reference[:2])
    try:model(images[:2],prototype_memory=bank)
    except ValueError:pass
    else:raise AssertionError('Inference accepted an identity memory')
    from .processor.processor_mldc import save_checkpoint
    from .export_inference import export
    with tempfile.TemporaryDirectory(prefix='pad_check_',dir=str(Path(args.report).resolve().parent)) as directory:
        cp=Path(directory)/'full.pth';ep=Path(directory)/'inference.pth'
        save_checkpoint(model,cfg,0,cp,memory=cnl,prototype_memory=bank)
        model.load_param(str(cp))
        with torch.no_grad():assert torch.equal(expected,model(images[:2]))
        export(cp,ep);model.load_param(str(ep))
        with torch.no_grad():assert torch.equal(expected,model(images[:2]))
        restored=PrototypeMemory(labels.tolist(),500,256);restored.load_state_dict(bank.state_dict())
        assert unchanged(bank.state_dict(),restored)
    result=dict(status='passed',optimizer_steps=0,training_launched=False,real_training_images=filenames,
        initial_baseline_max_abs_error=initial_error,p12_source_exact=True,attention_shape=[4,3,8,128],
        shared_layer_calls=counts,layers_have_independent_parameters=True,per_layer_per_head_kl_verified=True,
        kl_teacher_target_detached=True,prototype_query_has_identity_gradient=True,warmup_without_teacher=True,
        duplicate_image_counts_verified=True,forward_does_not_write_memory=True,
        amp_skip_does_not_write_either_memory=True,successful_step_updates_both_memories=True,
        final_bn_updated_once=True,cnl_uses_exact_final_bn_descriptor=True,gradient_norms=gradient,
        global_output_gradient_norm=float(gg.norm()),patch_output_gradient_norm=float(pg.norm()),
        low_precision_finite=True,checkpoint_roundtrip_exact=True,inference_export_retains_student=True,
        inference_rejects_prototype_memory=True,pad_parameters=sum(x.numel() for x in model.pad_head.parameters()),
        verification_note='Real-image gradients with frozen backbone; encoder-output connectivity checked separately. Memory commits are isolated fixtures, not optimizer training.')
    Path(args.report).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
