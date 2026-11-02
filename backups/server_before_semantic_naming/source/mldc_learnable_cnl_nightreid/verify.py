"""Verify the NightReID adaptation; all checks use zero optimizer steps."""
import argparse
import gc
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def configuration():
    from config import cfg as base_cfg
    from .model.make_model_mldc import add_mldc_config
    cfg=add_mldc_config(base_cfg.clone())
    cfg.merge_from_file(str(Path(__file__).parent/'configs/nightreid.yml'))
    return cfg


def seed():
    import numpy as np
    import torch
    random.seed(1234);np.random.seed(1234);torch.manual_seed(1234)


def digest(model):
    value=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        value.update(name.encode());value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def loaders(cfg, batch=4):
    from datasets import make_dataloader
    from .indexed_data import indexed_loader
    cfg=cfg.clone();cfg.defrost();cfg.DATALOADER.NUM_WORKERS=0
    cfg.SOLVER.IMS_PER_BATCH=batch
    if batch==4:cfg.DATALOADER.NUM_INSTANCE=2
    cfg.freeze()
    train,_,val,nq,nc,cameras,views=make_dataloader(cfg)
    indexed,dataset=indexed_loader(train)
    assert indexed.batch_sampler is train.batch_sampler
    return indexed,dataset,val,nq,nc,cameras,views


def cpu_check():
    import torch
    import torch.nn.functional as F
    from mldc_learnable_global_cnl.model.make_model_mldc import MLDCReID as Original,add_mldc_config
    from config import cfg as base_cfg
    from .model.make_model_mldc import MLDCReID,configure_head_learning_rates
    from .loss.make_loss_mldc import make_loss
    from .global_cnl import GlobalCNL
    from .verify_schedule import main as verify_schedule
    from .processor.visualization import fixed_pairs
    from .scheduler_mldc import create_fixed_equal_scheduler,wrap_weight_scheduler
    from solver import make_optimizer

    torch.set_num_threads(1);seed();cfg=configuration()
    original=add_mldc_config(base_cfg.clone())
    original.merge_from_file('mldc_learnable_global_cnl/variants/deep_four_uniform_cross_camera_cnl/config.yml')
    def flat(node,prefix=''):
        result={}
        for k,v in node.items():
            if hasattr(v,'items'):result.update(flat(v,prefix+k+'.'))
            else:result[prefix+k]=v
        return result
    old,new=flat(original),flat(cfg)
    differences={k:dict(source=old.get(k),nightreid=new.get(k)) for k in set(old)|set(new) if old.get(k)!=new.get(k)}
    assert set(differences)=={'DATASETS.NAMES','OUTPUT_DIR'},differences
    assert cfg.TEST.NECK_FEAT=='after' and cfg.SOLVER.MLDC_HEAD_LR_FACTOR==5.
    train,dataset,val,nq,nc,cameras,views=loaders(cfg)
    images,labels,camera,view,indices=next(iter(train))
    identity_cameras=defaultdict(set)
    for _,pid,cam,_ in dataset.entries:identity_cameras[pid].add(cam)
    assert len(identity_cameras)==nc and nq>0 and len(val.dataset)>nq
    cross_ids=sum(len(v)>1 for v in identity_cameras.values())
    assert cross_ids>0
    pairs=fixed_pairs(val.dataset.dataset,nq,cfg.TEST.MLDC_VIS_SAMPLES,1234)
    assert len(pairs)==cfg.TEST.MLDC_VIS_SAMPLES
    seed();baseline=Original(cfg,nc,cameras,views).eval()
    state=digest(baseline)
    with torch.no_grad():reference=baseline(images[:2]).clone()
    del baseline;gc.collect()
    seed();model=MLDCReID(cfg,nc,cameras,views).eval()
    assert digest(model)==state
    with torch.no_grad():assert torch.equal(model(images[:2]),reference)
    assert torch.equal(model.readout_weights().detach(),torch.tensor([0.]*8+[.25]*4))
    model.train()
    scores,raw,unit=model(images,labels,cam_label=camera,view_label=view,return_cnl_features=True)
    loss_fn,center=make_loss(cfg,nc)
    base_loss=loss_fn(scores,raw,labels,camera)
    bank=GlobalCNL([int(r[1]) for r in dataset.entries],[int(r[2]) for r in dataset.entries],
        gamma=cfg.SOLVER.CNL_GAMMA,temperature=cfg.SOLVER.CNL_TEMPERATURE,momentum=cfg.SOLVER.CNL_MOMENTUM)
    excluded=set(indices.tolist());anchor=None
    for j in range(4):
        positive=next((i for i,r in enumerate(dataset.entries) if i not in excluded and r[1]==int(labels[j]) and r[2]!=int(camera[j])),None)
        if positive is not None:anchor=j;break
    assert anchor is not None,'Real PK batch has no available cross-camera positive fixture'
    negative=next(i for i,r in enumerate(dataset.entries) if i not in excluded and r[1]!=int(labels[anchor]))
    direction=unit[anchor].detach().roll(1)
    keys=F.normalize(torch.stack([unit[anchor].detach()+.04*direction,unit[anchor].detach()-.04*direction]),dim=1)
    slots=torch.tensor([positive,negative]);bank.update(keys,slots,bank.labels[slots],bank.cameras[slots])
    saved=bank.features.clone();cnl,stats=bank(unit,indices,labels,camera)
    assert cnl.item()>0 and stats['active_anchors']>0
    logits_gradient=torch.autograd.grad(cnl,model.mldc.layer_weights,retain_graph=True)[0]
    assert torch.isfinite(logits_gradient).all() and logits_gradient.norm()>0
    total=base_loss+cfg.SOLVER.CNL_WEIGHT*cnl;total.backward()
    assert torch.equal(saved,bank.features)
    assert model.classifier.weight.shape==(nc,768)
    optimizer,_=make_optimizer(cfg,model,center);configure_head_learning_rates(cfg,model,optimizer)
    scheduler=wrap_weight_scheduler(cfg,model,optimizer,create_fixed_equal_scheduler(cfg,optimizer))
    names=dict(model.named_parameters())
    groups={id(g['params'][0]):g for g in optimizer.param_groups}
    initial={name:groups[id(names[name])]['lr'] for name in ['base.cls_token','classifier.weight','bottleneck.weight','mldc.layer_weights']}
    schedule=verify_schedule()
    return dict(all_passed=True,device='cpu',optimizer_steps=0,training_started=False,
        configuration_differences=differences,initial_state_and_descriptor_identical=True,
        training_images=len(dataset),training_classes=nc,training_cameras=cameras,
        training_identities_with_cross_camera_positives=cross_ids,queries=nq,gallery_images=len(val.dataset)-nq,
        dataset_fingerprint=dataset.fingerprint,query_dir='data/NightReID/query',gallery_dir='data/NightReID/bounding_box_test',
        visualization_pairs=len(pairs),descriptor_dim=768,initial_fusion_weights=[.25]*4,
        cnl_loss=float(cnl.detach()),cnl_stats=stats,cnl_only_fusion_logits_gradient=logits_gradient.tolist(),
        initial_learning_rates=initial,schedule_checks=schedule)


def cuda_check(batch):
    import torch
    from torch.cuda import amp
    from .model.make_model_mldc import MLDCReID
    from .loss.make_loss_mldc import make_loss
    from .global_cnl import GlobalCNL
    seed();cfg=configuration()
    train,dataset,_,_,nc,cameras,views=loaders(cfg,batch=batch)
    assert torch.cuda.is_available()
    free_before,total_memory=torch.cuda.mem_get_info()
    model=MLDCReID(cfg,nc,cameras,views).cuda().train()
    memory=GlobalCNL([int(r[1]) for r in dataset.entries],[int(r[2]) for r in dataset.entries],
        gamma=cfg.SOLVER.CNL_GAMMA,temperature=cfg.SOLVER.CNL_TEMPERATURE,momentum=cfg.SOLVER.CNL_MOMENTUM).cuda()
    # Fill the full-sized synthetic memory to exercise the eventual candidate count.
    with torch.no_grad():
        memory.features.copy_(torch.nn.functional.normalize(torch.randn_like(memory.features),dim=1));memory.valid.fill_(True)
    images,labels,camera,view,indices=[v.cuda() for v in next(iter(train))]
    loss_fn,_=make_loss(cfg,nc)
    torch.cuda.reset_peak_memory_stats()
    with amp.autocast():
        scores,raw,unit=model(images,labels,cam_label=camera,view_label=view,return_cnl_features=True)
        cnl,stats=memory(unit,indices,labels,camera)
        loss=loss_fn(scores,raw,labels,camera)+cfg.SOLVER.CNL_WEIGHT*cnl
    assert torch.isfinite(loss)
    loss.backward();torch.cuda.synchronize()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    reserved=torch.cuda.max_memory_reserved()
    momentum_bytes=sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
    capacity=reserved+momentum_bytes+512*1024**2 < free_before
    return dict(all_passed=bool(capacity),capacity_with_margin=bool(capacity),device='cuda',batch=batch,
        optimizer_steps=0,training_started=False,free_before=free_before,total_memory=total_memory,
        peak_reserved_bytes=reserved,estimated_momentum_bytes=momentum_bytes,loss=float(loss.detach()),cnl_stats=stats)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',required=True)
    parser.add_argument('--cuda-batch',type=int,default=0)
    args=parser.parse_args()
    result=cuda_check(args.cuda_batch) if args.cuda_batch else cpu_check()
    Path(args.report).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('VERIFICATION_RESULT '+json.dumps(result),flush=True)


if __name__=='__main__':main()
