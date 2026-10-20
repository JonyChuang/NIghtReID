"""Real SSL/ViT CPU verification, four real images, zero optimizer steps."""
import gc,json,random,ctypes
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .model import BRANCHES,add_config,config_from_dump,loss_weights,DualMLDCReID,retrieval_features,concatenate_unit_branches,objectives


def release():
    gc.collect()
    try:ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError,OSError):pass


def memory(label):
    path=Path('/sys/fs/cgroup/memory.stat')
    if not path.exists():return {}
    values=dict((a,int(b)) for a,b in (s.split() for s in path.read_text().splitlines()))
    result=dict(anonymous_bytes=values.get('anon',0),file_cache_bytes=values.get('file',0))
    print('VERIFY_MEMORY',label,result,flush=True)
    limit_path=Path('/sys/fs/cgroup/memory.max')
    limit_text=limit_path.read_text().strip() if limit_path.exists() else 'max'
    if limit_text!='max' and int(limit_text)<=3*1024**3 and result['anonymous_bytes']>1350*1024**2:
        raise MemoryError('Verification stopped before exhausting the small container')
    return result


def check_metric_contract():
    from .evaluate import evaluate_arrays
    from utils.metrics import eval_func
    g=torch.zeros(54,768);c=torch.zeros(54,768)
    g[:6,0:3]=torch.tensor([[1.,0,0],[0,1,0],[1,0,0],[1,.1,0],[.1,1,0],[0,0,1]])
    c[:6,0:3]=torch.tensor([[1.,0,0],[0,1,0],[1,0,0],[1,.2,0],[.2,1,0],[0,0,1]])
    g[6:,2]=1.;c[6:,2]=1.
    g[6,:3]=torch.tensor([1.,0,0]);c[6,:3]=torch.tensor([0.,0,1.])
    g*=torch.linspace(.2,7.,54)[:,None];c*=torch.linspace(4.,.5,54)[:,None]
    pid=[0,1,0,0,1,2]+[2]*48;cam=[0,0,0,1,1,1]+[1]*48
    raw=dict(fusion=g,last=c)
    neck=dict(fusion=g*.7,last=c*1.3)
    checked={}
    for position in ('before','after'):
        descriptors=retrieval_features(raw,neck,position)
        metrics,paired,rows=evaluate_arrays(descriptors['fusion'].numpy(),descriptors['last'].numpy(),pid,cam,2,chunk=1,
            concat=descriptors['concat'].numpy())
        for name,z in descriptors.items():
            dist=torch.cdist(z[:2],z[2:]).square().numpy()
            cmc,ap=eval_func(dist,np.array(pid[:2]),np.array(pid[2:]),np.array(cam[:2]),np.array(cam[2:]))
            assert abs(metrics[name]['mAP']-ap)<1e-8 and abs(metrics[name]['rank1']-cmc[0])<1e-6
        assert rows['concat'][0]['top1_gallery_index']!=0
        z=descriptors['concat'];gn=descriptors['fusion'];cn=descriptors['last']
        assert torch.allclose(z@z.T,.5*(gn@gn.T+cn@cn.T),atol=2e-6)
        assert torch.allclose(z[:,:768].square().sum(1),torch.full((54,),.5),atol=1e-6)
        assert torch.allclose(z[:,768:].square().sum(1),torch.full((54,),.5),atol=1e-6)
        scaled=dict(fusion=raw['fusion']*100,last=raw['last']*.001)
        assert torch.allclose(retrieval_features(scaled,neck,'before')['concat'],retrieval_features(raw,neck,'before')['concat'],atol=2e-6)
        checked[position]=dict(actual_descriptor_metric_matches_explicit=True,paired_contract=paired)
    # Random unequal-scale descriptors ensure a mean-distance implementation
    # would actually fail the AP comparison, rather than share trivial rankings.
    generator=torch.Generator().manual_seed(73)
    rg=torch.randn(54,768,generator=generator)*.1
    rc=torch.randn(54,768,generator=generator)*10.
    rz=concatenate_unit_branches(rg,rc)
    metrics,_,_=evaluate_arrays(rg.numpy(),rc.numpy(),pid,cam,2,chunk=1,concat=rz.numpy())
    explicit=torch.cdist(rz[:2],rz[2:]).square().numpy()
    _,actual_ap=eval_func(explicit,np.array(pid[:2]),np.array(pid[2:]),np.array(cam[:2]),np.array(cam[2:]))
    ng=F.normalize(rg,dim=1);nc=F.normalize(rc,dim=1)
    wrong_raw=F.normalize(torch.cat((rg,rc),dim=1),dim=1)
    wrong=torch.cdist(wrong_raw[:2],wrong_raw[2:]).square().numpy()
    _,wrong_ap=eval_func(wrong,np.array(pid[:2]),np.array(pid[2:]),np.array(cam[:2]),np.array(cam[2:]))
    assert abs(metrics['concat']['mAP']-actual_ap)<1e-8 and abs(actual_ap-wrong_ap)>.0001
    return dict(chunked_metric_matches_explicit_1536D=True,same_identity_same_camera_exclusion=True,
        rejects_raw_concat_substitution=True,concat_cosine_exact_mean=True,
        independent_source_scale_invariance=True,each_source_half_energy=True,readouts=checked)


def check_backward(model,images,labels,camera,view,cfg):
    from torch.utils.checkpoint import checkpoint
    from .processor import gradient_summary
    originals=[]
    # Verification-only activation recomputation keeps the full real ViT backward
    # within the CPU container limit. Training source retains normal block calls.
    for block in model.base.blocks:
        original=block.forward;originals.append((block,original))
        def wrapped(x,original=original):return checkpoint(original,x,use_reentrant=False,preserve_rng_state=True)
        block.forward=wrapped
    try:
        model.train();data=model(images,labels,cam_label=camera,view_label=view)
        loss,terms=objectives(data,labels,cfg)
        lg=terms['fusion_id']+terms['fusion_triplet'];lc=terms['last_id']+terms['last_triplet']
        wg,wc=loss_weights(cfg)
        assert set(terms)=={'fusion_id','fusion_triplet','last_id','last_triplet'}
        assert set(data['scores'])=={'fusion','last'}
        assert torch.allclose(loss,wg*lg+wc*lc,atol=1e-7)
        comparison=cfg.clone();comparison.defrost();comparison.MODEL.LAST_LOSS_WEIGHT=.5;comparison.freeze()
        loss05,_=objectives(data,labels,comparison)
        assert torch.allclose(loss05,.5*lg+.5*lc,atol=1e-7)
        targets=[model.classifier.weight,model.last_classifier.weight,model.base.norm.weight]
        gp=torch.autograd.grad(lg,targets,retain_graph=True,allow_unused=True)
        cp=torch.autograd.grad(lc,targets,retain_graph=True,allow_unused=True)
        assert gp[0] is not None and all(v is None for v in gp[1:])
        assert cp[0] is None and cp[1] is not None and cp[2] is not None and cp[2].abs().sum()>0
        memory('before_backward');loss.backward();usage=memory('after_backward')
        assert torch.allclose(model.classifier.weight.grad,wg*gp[0],atol=1e-6,rtol=1e-5)
        assert torch.allclose(model.last_classifier.weight.grad,wc*cp[1],atol=1e-6,rtol=1e-5)
        assert torch.allclose(model.base.norm.weight.grad,wc*cp[2],atol=2e-6,rtol=1e-5)
        missing=[name for name,p in model.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
        assert not missing,missing
        gradients=gradient_summary(model);assert all(v>0 for v in gradients.values()),gradients
        return dict(loss=float(loss),loss_at_last_weight05=float(loss05),fusion_loss_weight=wg,last_loss_weight=wc,
            weighted_fusion_loss=float(wg*lg),weighted_last_loss=float(wc*lc),
            terms={k:float(v) for k,v in terms.items()},gradient_norms=gradients,concat_loss_absent=True,
            independent_heads_and_LN_paths=True,configured_weight_gradients_exact=True,all_active_gradients_finite=True,
            activation_recomputation_for_verification_only=True,memory=usage)
    finally:
        for block,original in originals:block.forward=original
        model.zero_grad(set_to_none=True)


def optimizer_checks(cfg,model):
    from .optim import build_optimizer,build_scheduler,lr_summary
    optimizer=build_optimizer(cfg,model)
    ids=[id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad}
    rates={g['parameter_name']:g['lr'] for g in optimizer.param_groups}
    assert rates['classifier.weight']==rates['last_classifier.weight']==cfg.SOLVER.BASE_LR*5
    assert rates['bottleneck.weight']==rates['last_bottleneck.weight']
    scheduler=build_scheduler(cfg,optimizer);initial=lr_summary(optimizer)
    scheduler.step(20);epoch20=lr_summary(optimizer);scheduler.step(120);final=lr_summary(optimizer)
    assert not optimizer.state
    return dict(exact_optimizer_coverage=True,no_duplicate_parameters=True,start_lr=initial,epoch20_lr=epoch20,final_lr=final,optimizer_steps=0)


def main():
    from config import cfg as base_cfg
    from datasets import make_dataloader
    from .processor import save_checkpoint
    torch.set_num_threads(1);random.seed(1234);np.random.seed(1234);torch.manual_seed(1234)
    cfg=add_config(base_cfg.clone());cfg.merge_from_file(str(Path(__file__).with_name('variants/fixed_global_last_cls_unit_concat/config.yml')))
    replay=config_from_dump(base_cfg,cfg.dump());assert replay.dump()==cfg.dump()
    legacy='\n'.join(line for line in cfg.dump().splitlines() if 'LAST_LOSS_WEIGHT:' not in line)
    assert config_from_dump(base_cfg,legacy).MODEL.LAST_LOSS_WEIGHT==.5
    cfg.DATALOADER.NUM_WORKERS=0;cfg.TEST.IMS_PER_BATCH=4;cfg.freeze()
    metric=check_metric_contract()
    train_loader,_,_,_,nc,cameras,views=make_dataloader(cfg)
    model=DualMLDCReID(cfg,nc,cameras,views,initialize=True)
    assert all(p.requires_grad for block in model.base.blocks for p in block.parameters())
    assert all(p.requires_grad for p in model.base.norm.parameters())
    assert model.bottleneck is not model.last_bottleneck and model.classifier.weight.data_ptr()!=model.last_classifier.weight.data_ptr()
    assert not any(name.startswith('concat_') for name in model.state_dict())
    # Directly confirm trained ReID weights were not used as initialization.
    ssl=torch.load(cfg.MODEL.PRETRAIN_PATH,map_location='cpu',mmap=True,weights_only=False)
    for key in ['model','state_dict']:
        if key in ssl:ssl=ssl[key]
    for key in ['norm.weight','norm.bias','blocks.11.attn.qkv.weight','patch_embed.proj.weight']:
        assert torch.equal(model.base.state_dict()[key],ssl[key]),key
    del ssl;release()
    images,labels,camera,view=next(iter(train_loader))
    chosen=[]
    for identity in labels.unique()[:2]:chosen.extend(torch.where(labels==identity)[0][:2].tolist())
    assert len(chosen)==4
    images=images[chosen].clone();labels=labels[chosen].clone();camera=camera[chosen].clone();view=view[chosen].clone()
    del train_loader;release();memory('real_model_loaded')
    model.eval();calls=[0]*12
    hooks=[]
    for j,block in enumerate(model.base.blocks):
        def count(module,args,result,j=j):calls[j]+=1
        hooks.append(block.register_forward_hook(count))
    with torch.inference_mode():
        data=model(images,cam_label=camera,view_label=view,diagnostics=True)
        assert torch.equal(data['raw']['fusion'],data['layers'][8:12].mean(0))
        assert torch.equal(data['raw']['last'],model.base.norm(data['layers'][-1]))
        assert data['retrieval']['concat'].shape==(4,1536)
        assert torch.allclose(data['retrieval']['concat'].norm(dim=1),torch.ones(4),atol=1e-6)
        for position in ('before','after'):
            descriptors=retrieval_features(data['raw'],data['neck'],position)
            source=data['raw'] if position=='before' else data['neck']
            for name in ('fusion','last'):assert torch.allclose(descriptors[name],F.normalize(source[name].float(),dim=1),atol=1e-6)
            expected_concat=concatenate_unit_branches(source['fusion'],source['last'])
            assert torch.allclose(descriptors['concat'],expected_concat,atol=1e-6)
            kernel=descriptors['concat']@descriptors['concat'].T
            assert torch.allclose(kernel,.5*(descriptors['fusion']@descriptors['fusion'].T+descriptors['last']@descriptors['last'].T),atol=2e-6)
    for hook in hooks:hook.remove()
    assert calls==[1]*12,calls
    del data,source,descriptors,expected_concat,kernel;release()
    versions={n:p._version for n,p in model.named_parameters()}
    backward=check_backward(model,images,labels,camera,view,cfg);release()
    assert all(p._version==versions[n] for n,p in model.named_parameters())
    optimizer=optimizer_checks(cfg,model);release()
    model.eval()
    with torch.inference_mode():expected=model(images,cam_label=camera,view_label=view).clone()
    verification_dir=Path(__file__).with_name('verification_server_only');verification_dir.mkdir(exist_ok=True)
    path=verification_dir/'roundtrip.pth';save_checkpoint(path,model,cfg,0,None)
    architecture=model.architecture();del model;release();memory('before_reload')
    saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    restored=DualMLDCReID(cfg,nc,cameras,views,initialize=False)
    restored.load_state_dict(saved['state_dict'],strict=True,assign=True);restored.eval()
    with torch.inference_mode():actual=restored(images,cam_label=camera,view_label=view)
    assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-6)
    assert all(p.requires_grad for p in restored.base.norm.parameters())
    result=dict(variant=architecture['variant'],architecture=architecture,config_roundtrip=True,
        original_SSL_loaded_and_key_parameters_equal=True,real_training_images=4,real_training_identities=2,num_classes=nc,
        single_shared_backbone_pass=True,fusion_exact_fixed_mean=True,last_exact_pretrained_affine_LayerNorm=True,
        concat_dim=1536,source_wise_L2_before_concat=True,concat_cosine_exact_mean=True,
        concat_head_absent=True,before_and_after_readouts_verified=True,backward=backward,optimizer=optimizer,
        checkpoint_strict_roundtrip=True,metric_contract=metric,optimizer_steps=0,training_epochs=0,
        training_started=False,full_gallery_evaluated=False,CUDA_initialized=torch.cuda.is_initialized())
    (Path(__file__).with_name('verification.json')).write_text(json.dumps(result,indent=2))
    print('DUAL_SUPERVISION_VERIFIED',json.dumps(result),flush=True)
    return result


if __name__=='__main__':main()
