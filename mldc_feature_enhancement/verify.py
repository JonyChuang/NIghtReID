"""Server-only real-ViT checks; random inputs, no dataset or optimizer updates."""
import gc
import json
from pathlib import Path


def main():
    import torch
    from config import cfg as base_cfg
    from .model import add_config,config_from_dump,FeatEnHancerMLDC,FixedEqualMLDC,objectives
    from .patch_enhancer import PatchCompensation
    torch.set_num_threads(1)
    cfg=add_config(base_cfg.clone());cfg.merge_from_file(str(Path(__file__).with_name('variants/patch_fen_safa_residual/config.yml')));cfg.freeze()
    replay=config_from_dump(base_cfg,cfg.dump())
    assert replay.MODEL.DEVICE_ID==cfg.MODEL.DEVICE_ID and replay.dump()==cfg.dump()
    layers=torch.randn(12,4,768,requires_grad=True)
    mean=FixedEqualMLDC()(layers)
    assert torch.equal(mean,layers[8:12].mean(0))
    mean.sum().backward()
    assert not layers.grad[:8].any() and torch.equal(layers.grad[8:],torch.full_like(layers.grad[8:],.25))
    del layers,mean
    torch.manual_seed(1234)
    reader=PatchCompensation(768,192,64,[4,6,8],.1).eval()
    constant=torch.randn(3,2,1,768).expand(-1,-1,128,-1)
    g=torch.randn(2,768)
    with torch.no_grad():
        probe=reader(constant,g,(16,8),diagnostics=True)
        assert torch.allclose(probe['patch_attention'],torch.full_like(probe['patch_attention'],1/128),atol=1e-7)
        assert torch.allclose(probe['compensated'],g,atol=1e-5)
    del reader,constant,g,probe
    gc.collect()
    torch.manual_seed(1234)
    model=FeatEnHancerMLDC(cfg,4,0,0,initialize=False)
    assert isinstance(model.feat_enhancer,torch.nn.Identity)
    assert all(p.requires_grad for b in model.base.blocks for p in b.parameters())
    assert all(p.requires_grad for p in model.patch_compensation.parameters())
    initial_patch=model.base.patch_embed.proj.weight.detach().clone()
    initial_head=model.classifier.weight.detach().clone()
    inputs=torch.rand(2,3,256,128).mul(2).sub(1)
    model.eval()
    with torch.inference_mode():
        plain=model.base(inputs)
        selected,patches=model.base(inputs,patch_layers=(4,6,8))
        assert torch.equal(plain,selected) and patches.shape==(3,2,128,768)
        data=model(inputs,diagnostics=True)
        assert torch.equal(data['enhanced'],inputs)
        assert torch.equal(data['global_feature'],data['layers'][8:12].mean(0))
        assert data['patch_attention'].shape==(2,3,16,8)
        assert torch.allclose(data['patch_attention'].flatten(2).sum(-1),torch.ones(2,3),atol=1e-6)
        assert ((data['patch_attention'].flatten(2)-1/128).abs().sum(-1)>1e-4).all()
    del plain,selected,patches,data
    model.train()
    data=model(inputs)
    labels=torch.tensor([0,1])
    loss,id_loss,triplet=objectives(data,labels,cfg)
    assert torch.allclose(loss.detach(),id_loss+triplet)
    loss.backward()
    prefixes=['patch_compensation.projections.','patch_compensation.enhancer.fen.',
              'patch_compensation.enhancer.safa.query.','patch_compensation.enhancer.safa.key.',
              'patch_compensation.enhancer.output.','patch_compensation.query.','patch_compensation.key.',
              'patch_compensation.value.','patch_compensation.output.','base.patch_embed.','base.blocks.','classifier.']
    norms={prefix:sum(float(p.grad.float().square().sum()) for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None)**.5 for prefix in prefixes}
    assert all(v>0 and torch.isfinite(torch.tensor(v)) for v in norms.values()),norms
    missing=[n for n,p in model.patch_compensation.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or not p.grad.any()]
    assert not missing,missing
    count=sum(p.numel() for p in model.patch_compensation.parameters())
    architecture=model.architecture()
    model.zero_grad(set_to_none=True)
    del model,data,inputs,loss,id_loss,triplet
    gc.collect()
    torch.manual_seed(1234)
    control=cfg.clone();control.defrost();control.MODEL.FEATENHANCER_ENABLED=False;control.freeze()
    baseline=FeatEnHancerMLDC(control,4,0,0,initialize=False).eval()
    assert torch.equal(initial_patch,baseline.base.patch_embed.proj.weight)
    assert torch.equal(initial_head,baseline.classifier.weight)
    with torch.inference_mode():
        data=baseline(torch.zeros(1,3,256,128),diagnostics=True)
        assert torch.equal(data['raw_readout'],data['layers'][8:12].mean(0))
    del baseline,data,initial_patch,initial_head
    gc.collect()
    result=dict(scheme='B',architecture=architecture,config_roundtrip=True,
                raw_input_unchanged=True,descriptor_shape=[2,768],patch_shape=[3,2,128,768],
                collection_preserves_backbone_forward=True,fixed_equal_global_exact=True,
                constant_spatial_input_zero_compensation=True,nonuniform_attention_on_random_inputs=True,
                all_patch_parameters_have_finite_nonzero_gradients=True,gradient_norms=norms,
                patch_module_parameters=count,baseline_initialization_matches=True,
                full_id_plus_triplet=True,optimizer_steps=0,pretrained_checkpoint_loaded=False,
                dataset_evaluated=False,training_started=False)
    print('SCHEME_B_VERIFIED',json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    main()
