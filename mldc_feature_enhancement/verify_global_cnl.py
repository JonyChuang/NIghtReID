"""Check global-only architecture and real CNL math; backward only, no optimizer steps."""
import gc
import io
import json
import tempfile
from pathlib import Path


def check_neighbor_loss():
    import torch
    from torch.nn import functional as F
    from .global_cnl import GlobalCNL
    labels=torch.tensor([0,0,1,0,0]);cameras=torch.tensor([0,1,1,0,2]);indices=torch.arange(5)
    # The final same-ID/cross-camera image is below threshold: it must not be a negative.
    keys=F.normalize(torch.tensor([[1.,0.],[.98,.2],[1.,0.],[1.,0.],[.8,.6]]),dim=1)
    memory=GlobalCNL(labels,cameras,dim=2,temperature=.5)
    memory.update(keys,indices,labels,cameras)
    before={k:v.clone() for k,v in memory.state_dict().items()}
    query=torch.tensor([[1.,0.]],requires_grad=True)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        loss,stats=memory(query,indices[:1],labels[:1],cameras[:1])
    expected=F.softplus((1-keys[1,0])/.5)
    assert loss.dtype==torch.float32 and torch.allclose(loss,expected,atol=1e-6),(loss,expected)
    assert stats['selected_positives']==1 and stats['available_positives']==2 and stats['active_anchors']==1
    loss.backward();assert query.grad.abs().sum()>0 and torch.isfinite(query.grad).all()
    assert all(torch.equal(before[k],v) for k,v in memory.state_dict().items())
    assert memory.features.grad is None and not list(memory.parameters())
    # Self must be excluded before the maximum; otherwise this valid 0.8-positive is lost.
    threshold=GlobalCNL([0,0,1],[0,1,1],dim=2,temperature=.5)
    vectors=F.normalize(torch.tensor([[1.,0.],[.8,.6],[.2,.98]]),dim=1)
    threshold.update(vectors,torch.arange(3),threshold.labels,threshold.cameras)
    value,detail=threshold(vectors[:1],torch.tensor([0]),threshold.labels[:1],threshold.cameras[:1])
    assert value>0 and detail['selected_positives']==1
    # No selected cross-camera positive / no negative both yield a differentiable zero.
    for ys,cs in [([0,1],[0,1]),([0,0],[0,1])]:
        empty=GlobalCNL(ys,cs,dim=2)
        x=torch.tensor([[1.,0.],[1.,0.]],requires_grad=True)
        zero,detail=empty(x,torch.arange(2),empty.labels,empty.cameras)
        assert zero==0 and detail['active_anchors']==0
        zero.backward();assert torch.isfinite(x.grad).all()
    # PK duplicates update one slot, averaging detached directions once.
    duplicates=GlobalCNL([0,1],[0,1],dim=2)
    duplicates.update(torch.tensor([[1.,0.],[0.,1.]]),torch.tensor([0,0]),torch.tensor([0,0]),torch.tensor([0,0]))
    assert duplicates.valid.sum()==1
    assert torch.allclose(duplicates.features[0],F.normalize(torch.tensor([1.,1.]),dim=0))
    state=io.BytesIO();torch.save(memory.state_dict(),state);state.seek(0)
    restored=GlobalCNL(labels,cameras,dim=2,temperature=.5)
    restored.load_state_dict(torch.load(state,weights_only=True),strict=True)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in memory.state_dict().items())
    try:memory(query.detach(),torch.tensor([0]),torch.tensor([1]),torch.tensor([0]))
    except ValueError:pass
    else:raise AssertionError('Incorrect training-image labels accepted')
    return dict(analytic_loss=float(loss.detach()),positive_and_negative_gradient=True,
        identity_crosscamera_self_filter=True,unselected_same_id_not_negative=True,
        empty_and_no_negative_zero=True,fp32_under_autocast=True,
        one_bank_detached_targets=True,duplicate_image_mean_once=True,
        forward_does_not_update_memory=True,memory_roundtrip=True)


def main(initialize=True,real_data=True):
    import numpy as np
    import torch
    from torch import nn
    from config import cfg as base_cfg
    from .cppi_model import MLDCCNLReID,GLOBAL_CNL_VARIANT,add_config,config_from_dump
    from .cppi_optim import build_optimizer,build_scheduler
    from .cnn_sdcl import indexed_loader
    from .global_cnl import create_global_memory,global_training_loss
    from .cnn_sdcl_processor import evaluate,save_checkpoint
    torch.set_num_threads(1);torch.manual_seed(1234)
    cfg=add_config(base_cfg.clone());cfg.merge_from_file(str(Path(__file__).with_name('variants/global_cls_cross_camera_cnl/config.yml')))
    cfg.DATALOADER.NUM_WORKERS=0;cfg.SOLVER.IMS_PER_BATCH=4;cfg.DATALOADER.NUM_INSTANCE=2
    cfg.TEST.IMS_PER_BATCH=4;cfg.freeze()
    assert cfg.MODEL.ENHANCEMENT_MODE=='mldc_cnl' and not cfg.MODEL.CNN_SD_ENABLED
    assert cfg.SOLVER.CNL_WEIGHT==.1 and cfg.SOLVER.SHALLOW_LOSS_WEIGHT==0
    assert config_from_dump(base_cfg,cfg.dump()).dump()==cfg.dump()
    neighbor_checks=check_neighbor_loss()
    if real_data:
        from datasets import make_dataloader
        loader,_,_,_,nc,cameras,views=make_dataloader(cfg)
    else:
        from torch.utils.data import Dataset,DataLoader
        class Fixture(Dataset):
            dataset=[('fixture_'+str(i),i//2,i%2,0) for i in range(4)]
            def __len__(self):return 4
            def __getitem__(self,i):return torch.randn(3,256,128),i//2,i%2,0,self.dataset[i][0]
        loader=DataLoader(Fixture(),batch_size=4);nc,cameras,views=2,2,1
    sampler=loader.batch_sampler;loader,dataset=indexed_loader(loader)
    assert loader.batch_sampler is sampler
    images,labels,camera,view,indices=next(iter(loader))
    torch.manual_seed(1234)
    model=MLDCCNLReID(cfg,nc,cameras,views,initialize=initialize)
    assert not model.enabled and isinstance(model.feat_enhancer,nn.Identity)
    assert not any(name.startswith(('cppi_','cnn_reader.','shallow_','patch_compensation.')) for name,_ in model.named_parameters())
    assert all(p.requires_grad for block in model.base.blocks for p in block.parameters())
    model.eval()
    with torch.no_grad():
        data=model(images[:2],diagnostics=True)
        assert data['retrieval'].shape==(2,768) and not any(k.startswith('shallow') for k in data)
        assert torch.equal(data['global_raw'],data['layers'][8:12].mean(0))
        assert torch.allclose(data['retrieval'].norm(dim=1),torch.ones(2),atol=1e-6)
        initial_reference=data['retrieval'][:1].clone()
    del data
    optimizer=build_optimizer(cfg,model);scheduler=build_scheduler(cfg,optimizer)
    grouped=[id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(grouped)==len(set(grouped)) and set(grouped)=={id(p) for p in model.parameters() if p.requires_grad}
    assert {g['category'] for g in optimizer.param_groups}=={'backbone','head'}
    memory=create_global_memory(cfg,model,dataset,'cpu')
    assert memory.features.shape==(len(dataset),768) and not memory.valid.any()
    model.train()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        data=model(images,cam_label=camera,view_label=view)
        loss,detail,stats=global_training_loss(data,labels,cfg,memory,indices,camera)
    expected=cfg.MODEL.ID_LOSS_WEIGHT*detail['global_id']+cfg.MODEL.TRIPLET_LOSS_WEIGHT*detail['global_triplet']+cfg.SOLVER.CNL_WEIGHT*detail['cnl']
    assert abs(float(loss.detach())-expected)<1e-5
    assert not memory.valid.any()
    loss.backward()
    prefixes=['base.patch_embed.','base.blocks.','bottleneck.','classifier.']
    gradients={prefix:sum(float(p.grad.float().square().sum()) for n,p in model.named_parameters()
        if n.startswith(prefix) and p.grad is not None)**.5 for prefix in prefixes}
    assert all(np.isfinite(v) and v>0 for v in gradients.values()),gradients
    missing=[n for n,p in model.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not missing,missing
    # Explicit bank update exercises storage only, after the backward check; no parameter step.
    memory.update(data['global_unit'],indices,labels,camera)
    assert int(memory.valid.sum())==indices.unique().numel()
    detail['total']=float(loss.detach())
    del data,loss;model.zero_grad(set_to_none=True);del optimizer,scheduler;gc.collect()
    model.eval()
    with torch.no_grad():reference=model(images[:1])
    architecture=model.architecture();assert architecture['variant']==GLOBAL_CNL_VARIANT
    initial_stem=model.base.patch_embed.proj.weight.detach().clone();initial_head=model.classifier.weight.detach().clone()
    with tempfile.TemporaryDirectory(prefix='global_cnl_verify_',dir=str(Path(__file__).resolve().parent)) as folder:
        path=Path(folder)/'interface.pth';save_checkpoint(path,model,cfg,0,None,dataset.fingerprint,memory=memory)
        del model;gc.collect()
        saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        restored_cfg=config_from_dump(base_cfg,saved['config']);restored_cfg.freeze()
        clone=MLDCCNLReID(restored_cfg,nc,cameras,views,initialize=False).eval()
        clone.load_state_dict(saved['state_dict'],strict=True,assign=True)
        assert clone.architecture()==saved['architecture'] and saved['training_dataset_fingerprint']==dataset.fingerprint
        restored_memory=create_global_memory(restored_cfg,clone,dataset,'cpu')
        restored_memory.load_state_dict(saved['cnl_memory'],strict=True)
        assert all(torch.equal(v,restored_memory.state_dict()[k]) for k,v in memory.state_dict().items())
        with torch.no_grad():assert torch.equal(reference,clone(images[:1]))
        del clone,saved,restored_memory;gc.collect()
    control_cfg=add_config(base_cfg.clone());control_cfg.merge_from_file(str(Path(__file__).with_name('variants/global_cls_without_cnl/config.yml')));control_cfg.freeze()
    assert control_cfg.SOLVER.CNL_WEIGHT==0 and not control_cfg.MODEL.CNN_SD_ENABLED
    torch.manual_seed(1234);control=MLDCCNLReID(control_cfg,nc,cameras,views,initialize=initialize).eval()
    assert create_global_memory(control_cfg,control,dataset,'cpu') is None
    assert torch.equal(initial_stem,control.base.patch_embed.proj.weight) and torch.equal(initial_head,control.classifier.weight)
    # Compare before the verifier's train-mode pass changes BN running statistics.
    with torch.no_grad():assert torch.allclose(initial_reference,control(images[:1]),atol=1e-6,rtol=1e-5)
    del control;gc.collect()
    # Configuration recovery must retain historical CNN flags despite new defaults.
    from .cppi_model import CNNShallowDeepReID,CNN_FUSED_VARIANT,CNN_SDCL_VARIANT
    legacy=cfg.clone();legacy.defrost();legacy.MODEL.ENHANCEMENT_MODE='cnn_shallow_deep'
    legacy.MODEL.CNN_SD_ENABLED=True;legacy.SOLVER.CNL_WEIGHT=0.;legacy.MODEL.CNN_SD_QUERY='fused_cls'
    assert config_from_dump(base_cfg,legacy.dump()).MODEL.CNN_SD_ENABLED
    historical=legacy.clone();del historical.MODEL['CNN_SD_QUERY'];historical.SOLVER.CNL_WEIGHT=.1
    assert config_from_dump(base_cfg,historical.dump()).MODEL.CNN_SD_QUERY=='cls11'
    historical_model=CNNShallowDeepReID(legacy,nc,cameras,views,initialize=False).eval()
    assert historical_model.architecture()['variant']==CNN_FUSED_VARIANT
    historical_model.query_source='cls11';assert historical_model.architecture()['variant']==CNN_SDCL_VARIANT
    del historical_model;gc.collect()
    class GlobalFixture(nn.Module):
        enabled=False
        def forward(self,images,**kwargs):
            f=nn.functional.one_hot(images[:,0,0,0].long()%12,num_classes=12).float()
            return dict(global_unit=f,retrieval=f)
    ys=[0,1]+list(range(12));cs=[0,0]+[1]*12
    x=torch.zeros(14,3,1,1);x[:,0,0,0]=torch.tensor([0,1]+list(range(12)))
    fake=[(x[i:i+2],tuple(ys[i:i+2]),tuple(cs[i:i+2]),torch.zeros(2,dtype=torch.long),
        torch.zeros(2,dtype=torch.long),['fixture']*2) for i in range(0,14,2)]
    metrics,diagnostics=evaluate(cfg,GlobalFixture(),fake,2,'cpu')
    assert list(metrics)==['global'] and metrics['global']['rank1']==1
    assert diagnostics['primary_readout']=='global' and 'query_changes' not in diagnostics
    assert not torch.cuda.is_initialized()
    report=dict(architecture=architecture,cnl_checks=neighbor_checks,gradients=gradients,losses=detail,
        original_ssl_loaded=initialize,real_training_images=real_data,cpu_bfloat16_backward=True,
        no_cnn_reader_or_shallow_parameters=True,descriptor_dim=768,global_readout_evaluation=True,
        single_memory_bank=True,checkpoint_and_memory_roundtrip=True,matched_no_cnl_control=True,
        all_trainable_parameters_have_finite_gradients=True,optimizer_coverage_once=True,
        historical_cnn_configuration_preserved=True,indexed_loader_preserves_sampler=True,
        optimizer_steps=0,training_loop_started=False,cuda_initialized=False)
    print('GLOBAL_MLDC_CNL_VERIFIED',json.dumps(report),flush=True)
    return report


if __name__=='__main__':main()
