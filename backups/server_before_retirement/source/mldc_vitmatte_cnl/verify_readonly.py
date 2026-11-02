"""Default verification performs inference only and never updates parameters."""
import gc
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from .model.base_model import MLDCReID as Baseline
from .model.make_model_mldc import MLDCReID,add_mldc_config
from .model.detail_branch import ViTMatteDetails


def run(report_path):
    from config import cfg as original
    from .verify import real_batch,reset_seed
    from .model.vitmatte_official import Detail_Capture
    reset_seed();torch.set_num_threads(1);started=time.monotonic()
    package=Path(__file__).resolve().parent
    configs={}
    for name in ['dcm','baseline','dcm_mean']:
        cfg=add_mldc_config(original.clone());cfg.merge_from_file(str(package/'configs'/(name+'.yml')));cfg.freeze()
        assert cfg.SOLVER.SEED==1234
        configs[name]=cfg
    assert not configs['baseline'].MODEL.DETAIL_ENABLED
    assert configs['dcm_mean'].MODEL.DETAIL_POOLING=='mean'
    cfg=configs['dcm'];images,labels,cameras,classes=real_batch(cfg)
    print('READONLY_CHECK official DCM and mean-pooling control',flush=True)
    with torch.inference_mode():
        guided=ViTMatteDetails().eval();official=Detail_Capture(in_chans=768,img_chans=3).eval()
        copied={k:v for k,v in guided.state_dict().items() if k.startswith(('convstream.','fusion_blks.'))}
        official.load_state_dict(copied,strict=False)
        spatial=torch.randn(2,768,4,2);rgb=images[:2,:,:64,:32]
        features=spatial;details=official.convstream(rgb)
        for i,block in enumerate(official.fusion_blks):features=block(features,details['D'+str(3-i)])
        assert torch.equal(features,guided.capture(spatial,rgb))
        mean=ViTMatteDetails(pooling='mean').eval()
        desc=mean(spatial.flatten(2).transpose(1,2),rgb,torch.randn(2,768))
        assert desc.shape==(2,256) and torch.isfinite(desc).all()
        assert not any(p.requires_grad for p in mean.query.parameters())
    del guided,official,copied,spatial,rgb,features,details,mean,desc;gc.collect()
    checkpoint=Path('/root/log/transreid/night600/mldc_learnable_global_cnl/deep_four_uniform_cross_camera_cnl_fixed_equal_schedule/transformer_120.pth')
    before=checkpoint.stat()
    print('READONLY_CHECK current global checkpoint reference',flush=True)
    baseline=Baseline(cfg,classes,8,1,initialize=False).eval()
    weights=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    state={(k[7:] if k.startswith('module.') else k):v for k,v in weights['state_dict'].items()}
    baseline.load_state_dict(state,strict=True,assign=True)
    with torch.inference_mode():expected=F.normalize(baseline(images[:2]),dim=1).clone()
    del weights,state,baseline;gc.collect()
    model=MLDCReID(cfg,classes,8,1,initialize=False).eval()
    model.load_global_checkpoint(checkpoint)
    versions={k:p._version for k,p in model.named_parameters()}
    buffers={k:b.clone() for k,b in model.named_buffers()}
    with torch.inference_mode():
        data=model(images[:2],return_details=True,return_maps=True)
        assert torch.equal(data['global_unit'],expected)
        assert data['fused_unit'].shape==(2,1024)
        assert torch.isfinite(data['fused_unit']).all()
        assert torch.allclose(data['fused_unit'].norm(dim=1),torch.ones(2),atol=1e-6)
        assert torch.allclose(data['fused_unit']@data['fused_unit'].T,
            .9*(data['global_unit']@data['global_unit'].T)+.1*(data['detail_unit']@data['detail_unit'].T),atol=1e-6)
        model.detail_enabled=False
        assert torch.equal(model(images[:2]),expected)
        model.detail_enabled=True
    assert all(p._version==versions[k] and p.grad is None for k,p in model.named_parameters())
    assert all(torch.equal(b,buffers[k]) for k,b in model.named_buffers())
    after=checkpoint.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    report=dict(seed=1234,device='cpu',all_passed=True,mode='inference_only',optimizer_steps=0,
        full_training_started=False,retrieval_performance_measured=False,original_checkpoint_unchanged=True,
        model_parameters_unchanged=True,model_buffers_unchanged=True,official_rgb_capture_exact=True,
        config_variants=['dcm','baseline','dcm_mean'],mean_pooling_forward=True,
        trained_global_import_exact=True,disabled_control_exact=True,cosine_mixture_exact=True,
        detail_shapes=data['maps']['detail_shapes'],fusion_shapes=data['maps']['fusion_shapes'],
        descriptor_dim=model.descriptor_dim,architecture=model.architecture(),
        added_parameters=sum(p.numel() for n,p in model.named_parameters() if n.startswith(
            ('detail_branch.','detail_bottleneck.','fused_bottleneck.','fused_classifier.'))),
        gpu_batch64_validated=False,seconds=time.monotonic()-started)
    Path(report_path).write_text(json.dumps(report,indent=2)+'\n')
    print('VERIFICATION_RESULT '+json.dumps(report),flush=True)
