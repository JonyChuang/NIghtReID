"""Verify the isolated learned-fusion CNL integration; no optimizer steps."""
import gc,hashlib,json,random,tempfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

def digest(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def seed():
    random.seed(1234);np.random.seed(1234);torch.manual_seed(1234)

def main():
    from config import cfg as base_cfg
    from datasets import make_dataloader
    from solver import make_optimizer
    from solver.scheduler_factory import create_scheduler
    from mldc_deep4_readout.model.make_model_mldc import MLDCReID as Baseline,add_mldc_config as old_config,configure_head_learning_rates as old_rates
    from mldc_deep4_readout.loss.make_loss_mldc import make_loss as old_loss
    from mldc_deep4_readout.scheduler_mldc import wrap_weight_scheduler as old_wrap
    from .model.make_model_mldc import MLDCReID,add_mldc_config,configure_head_learning_rates
    from .loss.make_loss_mldc import make_loss
    from .scheduler_mldc import create_fixed_equal_scheduler,wrap_weight_scheduler
    from .verify_schedule import main as verify_schedule
    from .indexed_data import indexed_loader
    from .global_cnl import GlobalCNL
    from .processor.processor_mldc import save_checkpoint
    torch.set_num_threads(1)
    cfg=add_mldc_config(base_cfg.clone());cfg.merge_from_file(str(Path(__file__).parent/'variants/deep_four_uniform_cross_camera_cnl/config.yml'))
    oldcfg=old_config(base_cfg.clone());oldcfg.merge_from_file('mldc_deep4_readout/variants/deep_four_uniform_learnable/config.yml')
    def flatten(node,prefix=''):
        values={}
        for key,value in node.items():
            if hasattr(value,'items'):values.update(flatten(value,prefix+key+'.'))
            else:values[prefix+key]=value
        return values
    before=flatten(oldcfg);after=flatten(cfg)
    diffs={k:dict(baseline=before.get(k),new=after.get(k)) for k in set(before)|set(after) if before.get(k)!=after.get(k)}
    assert set(diffs)=={'OUTPUT_DIR','SOLVER.CNL_WEIGHT','SOLVER.CNL_GAMMA','SOLVER.CNL_TEMPERATURE','SOLVER.CNL_MOMENTUM'},diffs
    cfg.DATALOADER.NUM_WORKERS=0;cfg.SOLVER.IMS_PER_BATCH=4;cfg.DATALOADER.NUM_INSTANCE=2;cfg.TEST.IMS_PER_BATCH=4
    seed();loader,_,val,nq,nc,cameras,views=make_dataloader(cfg)
    sampler=loader.batch_sampler;loader,dataset=indexed_loader(loader);assert loader.batch_sampler is sampler
    # Real PK images exercise original ID+Triplet; the neighbor fixture below
    # deliberately guarantees an active CNL relation for a gradient check.
    images,labels,camera,view,indices=next(iter(loader))
    def optimizer_and_schedule(config,model,loss_builder,rates,wrap,scheduler_builder=create_scheduler):
        loss,center=loss_builder(config,nc);optimizer,_=make_optimizer(config,model,center)
        rates(config,model,optimizer);scheduler=wrap(config,model,optimizer,scheduler_builder(config,optimizer))
        initial=[float(g['lr']) for g in optimizer.param_groups]
        trajectory={0:initial}
        for epoch in [1,20,40,119]:
            scheduler.step(epoch);trajectory[epoch]=[float(g['lr']) for g in optimizer.param_groups]
        return loss,optimizer,trajectory
    seed();baseline=Baseline(oldcfg,nc,cameras,views).eval()
    original_digest=digest(baseline)
    with torch.no_grad():reference=baseline(images[:2]).clone()
    _,old_optimizer,old_trajectory=optimizer_and_schedule(oldcfg,baseline,old_loss,old_rates,old_wrap)
    old_trainable=[n for n,p in baseline.named_parameters() if p.requires_grad]
    del baseline,old_optimizer;gc.collect()
    seed();model=MLDCReID(cfg,nc,cameras,views).eval()
    assert digest(model)==original_digest
    with torch.no_grad():assert torch.equal(reference,model(images[:2]))
    assert [n for n,p in model.named_parameters() if p.requires_grad]==old_trainable
    loss_fn,optimizer,trajectory=optimizer_and_schedule(cfg,model,make_loss,configure_head_learning_rates,wrap_weight_scheduler,create_fixed_equal_scheduler)
    schedule_report=verify_schedule()
    assert trajectory!=old_trajectory
    groups=[g for g in optimizer.param_groups if any(p is model.mldc.layer_weights for p in g['params'])]
    assert len(groups)==1 and len(groups[0]['params'])==1
    assert torch.equal(model.readout_weights().detach(),torch.tensor([0.]*8+[.25]*4))
    assert model.mldc.layer_weights.requires_grad and model.mldc.layer_weights.numel()==4
    bank=GlobalCNL([int(r[1]) for r in dataset.entries],[int(r[2]) for r in dataset.entries],dim=768)
    model.train()
    scores,raw,unit=model(images,labels,cam_label=camera,view_label=view,return_cnl_features=True)
    assert scores.shape==(4,nc) and raw.shape==unit.shape==(4,768)
    assert torch.allclose(unit.norm(dim=1),torch.ones(4),atol=1e-6)
    base_loss=loss_fn(scores,raw,labels,camera)
    # Choose a genuine training-image slot of the same identity in another
    # camera, and a different-ID slot. Synthetic nearby keys test connectivity
    # only; these are not performance measurements or saved training state.
    idxset=set(indices.tolist());anchor=None
    for j in range(4):
        positive=next((i for i,r in enumerate(dataset.entries) if i not in idxset and r[1]==int(labels[j]) and r[2]!=int(camera[j])),None)
        if positive is not None:anchor=j;break
    assert anchor is not None
    negative=next(i for i,r in enumerate(dataset.entries) if i not in idxset and r[1]!=int(labels[anchor]))
    direction=torch.roll(unit[anchor].detach(),1)
    keys=F.normalize(torch.stack([unit[anchor].detach()+.04*direction,unit[anchor].detach()-.04*direction]),dim=1)
    slots=torch.tensor([positive,negative]);bank.update(keys,slots,bank.labels[slots],bank.cameras[slots])
    saved_memory={k:v.clone() for k,v in bank.state_dict().items()}
    cnl,stats=bank(unit,indices,labels,camera)
    assert cnl>0 and stats['active_anchors']>0
    cnl_logits_grad=torch.autograd.grad(cnl,model.mldc.layer_weights,retain_graph=True)[0]
    assert torch.isfinite(cnl_logits_grad).all() and cnl_logits_grad.abs().sum()>1e-10
    loss=base_loss+cfg.SOLVER.CNL_WEIGHT*cnl;loss.backward()
    assert all(torch.equal(v,bank.state_dict()[k]) for k,v in saved_memory.items())
    missing=[n for n,p in model.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not missing,missing
    gradient={prefix:sum(float(p.grad.square().sum()) for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None)**.5 for prefix in ['mldc.','base.blocks.','base.patch_embed.','bottleneck.','classifier.']}
    assert all(v>0 for v in gradient.values())
    terms=dict(identity=float(loss_fn.last_terms['identity']),triplet=float(loss_fn.last_terms['triplet']),cnl=float(cnl.detach()),total=float(loss.detach()))
    model.zero_grad(set_to_none=True);model.eval()
    with torch.no_grad():expected=model(images[:1]).clone()
    state_digest=digest(model)
    with tempfile.TemporaryDirectory(prefix='verification_',dir=str(Path(__file__).parent)) as directory:
        path=Path(directory)/'roundtrip.pth'
        save_checkpoint(model,cfg,0,path,memory=bank,dataset_fingerprint=dataset.fingerprint)
        del model,optimizer,raw,unit,scores,loss,base_loss,cnl;gc.collect()
        saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        restore=CfgNode_load(saved['config']);restore.MODEL.PRETRAIN_CHOICE='self';restore.MODEL.PRETRAIN_PATH=str(path)
        class LoadedOnce(MLDCReID):
            def load_param(self,path):pass
        clone=LoadedOnce(restore,nc,cameras,views).eval()
        clone.load_state_dict(saved['state_dict'],strict=True,assign=True)
        assert digest(clone)==state_digest and saved['cnl_memory']['features'].shape==bank.features.shape
        with torch.no_grad():assert torch.equal(expected,clone(images[:1]))
        del clone,saved;gc.collect()
    report=dict(variant='mldc_learnable_cnl_20261020',configuration_differences=diffs,
        baseline_initial_state_identical=True,initial_descriptor_identical=True,
        original_optimizer_groups_retained=True,lr_schedule='fixed_equal_group_scaled_cosine',
        schedule_checks=schedule_report,original_trainable_parameters_identical=True,
        initial_layer_weights=[.25]*4,learned_fusion_logits=4,descriptor_dim=768,
        training_images=len(dataset),training_dataset_fingerprint=dataset.fingerprint,
        cnl_fixture_scope='Real training images and metadata; controlled nearby detached memory keys to guarantee a nonzero CNL gradient, not a retrieval result',
        cnl_only_fusion_logits_gradient=cnl_logits_grad.tolist(),gradient_norms=gradient,loss_terms=terms,
        cnl_forward_does_not_update_memory=True,checkpoint_model_and_memory_roundtrip=True,
        cuda_available=torch.cuda.is_available(),optimizer_steps=0,training_loop_started=False)
    print('VERIFICATION_RESULT',json.dumps(report),flush=True)
    return report

def CfgNode_load(text):
    from yacs.config import CfgNode
    return CfgNode.load_cfg(text)

if __name__=='__main__':main()
