"""Critical integration checks with original SSL and four real training images."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import re
import time
import torch
from torch.nn import functional as F
from .model.base_model import MLDCReID as Baseline
from .model.make_model_mldc import MLDCReID,add_mldc_config,configure_head_learning_rates,fuse_units
from .model.detail_branch import ViTMatteDetails
from .model.vitmatte_official import Detail_Capture
from .loss.make_loss_mldc import make_loss
from .scheduler_mldc import create_fixed_equal_scheduler,wrap_weight_scheduler
from .global_cnl import GlobalCNL
from .objectives import supervised_losses
from .processor.processor_mldc import save_checkpoint


def reset_seed():
    import numpy as np
    random.seed(1234);np.random.seed(1234);torch.manual_seed(1234)


def digest(model):
    value=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if name.startswith(('detail_branch.','detail_bottleneck.','fused_bottleneck.','fused_classifier.')):continue
        value.update(name.encode());value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def real_batch(cfg):
    import numpy as np
    from PIL import Image
    groups={}
    for p in sorted(Path('data/night600/bounding_box_train').glob('*.jpg')):
        match=re.match(r'(\d{6})_c([1-8])_',p.name)
        if match and int(match[1])>0:
            groups.setdefault(int(match[1]),{}).setdefault(int(match[2])-1,[]).append(p)
    eligible=sorted(p for p,c in groups.items() if len(c)>1)
    rng=np.random.RandomState(1234)
    chosen=[eligible[i] for i in rng.permutation(len(eligible))[:2]]
    relabel={p:i for i,p in enumerate(sorted(groups))}
    images,labels,cameras=[],[],[]
    for pid in chosen:
        for cam in sorted(groups[pid])[:2]:
            paths=groups[pid][cam];path=paths[int(rng.randint(len(paths)))]
            with Image.open(path) as im:
                arr=np.asarray(im.convert('RGB').resize((128,256),Image.Resampling.BICUBIC)).copy()
            images.append(torch.from_numpy(arr).permute(2,0,1).float()/255)
            labels.append(relabel[pid]);cameras.append(cam)
    images=torch.stack(images)
    mean=torch.tensor(cfg.INPUT.PIXEL_MEAN)[None,:,None,None]
    std=torch.tensor(cfg.INPUT.PIXEL_STD)[None,:,None,None]
    return (images-mean)/std,torch.tensor(labels),torch.tensor(cameras),len(groups)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',required=True)
    parser.add_argument('--allow-update',action='store_true',help='Explicitly allow one temporary integration-test optimizer update')
    args=parser.parse_args()
    if not args.allow_update:
        from .verify_readonly import run
        run(args.report)
        return
    started=time.monotonic();torch.set_num_threads(1);reset_seed()
    from config import cfg as original
    from solver import make_optimizer
    cfg=add_mldc_config(original.clone())
    cfg.merge_from_file(str(Path(__file__).resolve().parent/'configs/dcm.yml'));cfg.freeze()
    images,labels,cameras,classes=real_batch(cfg)
    report=dict(seed=1234,device='cpu',real_training_images=len(images),identities=2,
        cameras=cameras.tolist(),full_training_started=False,verification_optimizer_steps=0)
    print('VERIFY official module equivalence',flush=True)
    branch=ViTMatteDetails().eval()
    reference=Detail_Capture(in_chans=768,img_chans=3).eval()
    state={k:v for k,v in branch.state_dict().items() if k.startswith(('convstream.','fusion_blks.'))}
    result=reference.load_state_dict(state,strict=False)
    assert all(k.startswith('matting_head.') for k in result.missing_keys) and not result.unexpected_keys
    with torch.inference_mode():
        spatial=torch.randn(2,768,4,2);rgb=images[:2,:,:64,:32]
        actual=branch.capture(spatial,rgb)
        maps=reference.convstream(rgb);expected=spatial
        for i,block in enumerate(reference.fusion_blks):expected=block(expected,maps['D'+str(3-i)])
        assert torch.equal(actual,expected)
    report['official_rgb_capture_exact']=True
    del branch,reference,state,actual,expected,maps;gc.collect()
    print('VERIFY original SSL baseline initialization',flush=True)
    reset_seed();baseline=Baseline(cfg,classes,8,1).eval()
    initial_digest=digest(baseline);baseline_rng=torch.random.get_rng_state().clone()
    with torch.inference_mode():base_output=F.normalize(baseline(images[:2]),dim=1).clone()
    del baseline;gc.collect()
    reset_seed();model=MLDCReID(cfg,classes,8,1).eval()
    assert digest(model)==initial_digest
    assert torch.equal(torch.random.get_rng_state(),baseline_rng)
    with torch.inference_mode():
        details=model(images[:2],return_details=True,return_maps=True)
        assert torch.equal(base_output,details['global_unit'])
        assert details['fused_unit'].shape==(2,1024) and details['detail_unit'].shape==(2,256)
        assert torch.allclose(details['fused_unit'].norm(dim=1),torch.ones(2),atol=1e-6)
        assert torch.allclose(details['fused_unit']@details['fused_unit'].T,
            .9*(details['global_unit']@details['global_unit'].T)+.1*(details['detail_unit']@details['detail_unit'].T),atol=1e-6)
        report['detail_shapes']=details['maps']['detail_shapes'];report['fusion_shapes']=details['maps']['fusion_shapes']
    report.update(global_initialization_exact=True,baseline_rng_preserved=True,global_eval_output_exact=True,
        cosine_mixture_exact=True,descriptor_dim=1024,architecture=model.architecture(),
        added_parameters=sum(p.numel() for n,p in model.named_parameters() if n.startswith(
            ('detail_branch.','detail_bottleneck.','fused_bottleneck.','fused_classifier.'))))
    del details,base_output;gc.collect()
    # Independent disabled control recovers the original forward exactly.
    model.detail_enabled=False
    with torch.inference_mode():disabled=model(images[:2]);disabled_details=model(images[:2],return_details=True)
    assert disabled.shape==(2,768) and torch.equal(disabled,disabled_details['global_unit'])
    model.detail_enabled=True;report['disabled_control_exact']=True
    del disabled,disabled_details
    print('VERIFY full joint backward and actual optimizer coverage',flush=True)
    loss_fn,center=make_loss(cfg,classes)
    optimizer,_=make_optimizer(cfg,model,center);configure_head_learning_rates(cfg,model,optimizer)
    scheduler=wrap_weight_scheduler(cfg,model,optimizer,create_fixed_equal_scheduler(cfg,optimizer))
    expected_ids={id(p) for p in model.parameters() if p.requires_grad}
    actual_ids=[id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(actual_ids)==len(set(actual_ids)) and set(actual_ids)==expected_ids
    lr={n:next(g['lr'] for g in optimizer.param_groups if any(p is v for p in g['params']))
        for n,v in model.named_parameters() if n in ['base.cls_token','mldc.layer_weights','classifier.weight',
            'detail_branch.convstream.convs.0.conv.weight','fused_classifier.weight']}
    assert abs(lr['base.cls_token']-lr['mldc.layer_weights'])<1e-12
    assert abs(lr['detail_branch.convstream.convs.0.conv.weight']-lr['base.cls_token'])<1e-12
    assert abs(lr['fused_classifier.weight']/lr['base.cls_token']-5)<1e-6
    model.train();optimizer.zero_grad(set_to_none=True)
    memory=GlobalCNL(labels,cameras,dim=768,gamma=cfg.SOLVER.CNL_GAMMA,
        temperature=cfg.SOLVER.CNL_TEMPERATURE,momentum=cfg.SOLVER.CNL_MOMENTUM)
    indices=torch.arange(len(images))
    with torch.autocast('cpu',dtype=torch.bfloat16):
        data=model(images,return_training=True)
        supervised,terms=supervised_losses(data,labels,cameras,loss_fn,cfg)
        cnl,stats=memory(data['global_unit'],indices,labels,cameras)
        total=supervised+cfg.SOLVER.CNL_WEIGHT*cnl
    total.backward()
    missing=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing,missing
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    prefixes=['base.patch_embed.','base.blocks.','mldc.','detail_branch.spatial_norm.',
        'detail_branch.convstream.','detail_branch.fusion_blks.','detail_branch.query.',
        'detail_branch.project.','detail_bottleneck.','fused_classifier.']
    gradients={prefix:sum(float(p.grad.float().norm()) for n,p in model.named_parameters()
        if n.startswith(prefix) and p.grad is not None) for prefix in prefixes}
    assert all(value>0 for value in gradients.values()),gradients
    watched=model.detail_branch.convstream.convs[0].conv.weight
    before=watched.detach().clone();optimizer.step()
    assert not torch.equal(before,watched.detach())
    memory.update(data['global_unit'],indices,labels,cameras)
    report.update(full_joint_backward=True,all_active_gradients_finite=True,gradient_norms=gradients,
        optimizer_coverage_exact=True,initial_learning_rates=lr,loss_terms=terms,cnl=float(cnl.detach()),
        cnl_stats=stats,cnl_memory_images=int(memory.valid.sum()),verification_optimizer_steps=1,
        cnn_parameters_updated=True)
    optimizer.zero_grad(set_to_none=True)
    del data,supervised,cnl,total,optimizer,scheduler,center,loss_fn,before,watched;gc.collect()
    print('VERIFY strict checkpoint restore',flush=True)
    model.eval()
    with torch.inference_mode():expected_output=model(images[:2]).clone()
    temporary=Path(args.report).with_suffix('.roundtrip.pth')
    save_checkpoint(model,cfg,1,temporary,memory)
    del model,memory;gc.collect()
    restored_cfg=cfg.clone();restored_cfg.defrost();restored_cfg.MODEL.PRETRAIN_CHOICE='self'
    restored_cfg.MODEL.PRETRAIN_PATH=str(temporary);restored_cfg.freeze()
    model=MLDCReID(restored_cfg,classes,8,1).eval()
    with torch.inference_mode():restored_output=model(images[:2])
    assert torch.equal(expected_output,restored_output)
    report['checkpoint_roundtrip_exact']=True
    temporary.unlink()
    del expected_output,restored_output;gc.collect()
    print('VERIFY current trained global checkpoint import',flush=True)
    original_checkpoint=Path('/root/log/transreid/night600/mldc_learnable_global_cnl/deep_four_uniform_cross_camera_cnl_fixed_equal_schedule/transformer_120.pth')
    before=original_checkpoint.stat()
    model.load_global_checkpoint(original_checkpoint)
    with torch.inference_mode():global_import=model(images[:2],return_details=True)['global_unit'].clone()
    del model;gc.collect()
    checkpoint=torch.load(original_checkpoint,map_location='cpu',mmap=True,weights_only=False)
    baseline=Baseline(cfg,classes,8,1,initialize=False)
    baseline.load_state_dict(checkpoint['state_dict'],strict=True,assign=True);baseline.eval()
    with torch.inference_mode():old_global=F.normalize(baseline(images[:2]),dim=1)
    assert torch.equal(global_import,old_global)
    after=original_checkpoint.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    report.update(trained_global_import_exact=True,original_checkpoint_unchanged=True,
        full_training_started=False,retrieval_performance_measured=False,
        gpu_batch64_validated=False,all_passed=True,seconds=time.monotonic()-started)
    Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
    print('VERIFICATION_RESULT '+json.dumps(report),flush=True)


if __name__=='__main__':main()
