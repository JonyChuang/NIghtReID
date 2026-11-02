"""Forward/compatibility checks and a small synthetic gradient check; zero updates."""
import ast
import gc
import hashlib
import json
from pathlib import Path
import time
import types
import torch
from torch import nn
from torch.nn import functional as F
from .model.make_model_mldc import MLDCReID,add_mldc_config,configure_head_learning_rates
from .model.conv_neck import ResidualConvNeck
from .model.backbones.vit_pytorch_mldc import Block


def common_digest(model):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if '.conv_neck.' in name:continue
        digest.update(name.encode());digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def verify_neck(package):
    # Execute the pinned upstream classes, replacing framework wrappers only.
    class CNNBlockBase(nn.Module):
        def __init__(self,in_channels,out_channels,stride):
            super().__init__()
    def msra(layer):
        nn.init.kaiming_normal_(layer.weight,mode='fan_out',nonlinearity='relu')
    tree=ast.parse((package/'references/vitmatte_vit.py.txt').read_text())
    classes=[node for node in tree.body if isinstance(node,ast.ClassDef)
             and node.name in ('LayerNorm','ResBottleneckBlock')]
    namespace=dict(torch=torch,nn=nn,F=F,CNNBlockBase=CNNBlockBase,Conv2d=nn.Conv2d,
                   weight_init=types.SimpleNamespace(c2_msra_fill=msra))
    namespace['get_norm']=lambda name,channels:namespace['LayerNorm'](channels,data_format='channels_first')
    exec(compile(ast.Module(body=classes,type_ignores=[]),'<pinned ViTMatte classes>','exec'),namespace)
    neck=ResidualConvNeck(32).eval()
    reference=namespace['ResBottleneckBlock'](32,32,16).eval()
    activated={k:v.detach().clone() for k,v in neck.state_dict().items()}
    activated['norm3.weight'].fill_(.1)
    reference.load_state_dict(activated,strict=True)
    activated_neck=ResidualConvNeck(32).eval()
    activated_neck.load_state_dict(activated,strict=True)
    spatial=torch.randn(2,32,4,2)
    tokens=torch.randn(2,9,32)
    with torch.inference_mode():
        assert torch.equal(neck(spatial),spatial)
        assert torch.equal(reference(spatial),activated_neck(spatial))
        updated=activated_neck.forward_tokens(tokens,(4,2))
        assert torch.equal(updated[:,:1],tokens[:,:1])
        assert not torch.equal(updated[:,1:],tokens[:,1:])
    # At identity initialization the last norm learns first, as in upstream.
    neck(spatial).square().mean().backward()
    assert neck.norm3.weight.grad.norm()>0
    assert neck.conv1.weight.grad.norm()==0
    activated_neck(spatial).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in activated_neck.parameters())
    assert all(p.grad.norm()>0 for p in activated_neck.parameters())
    # Pre-attention can affect this Block's CLS; post-MLP needs the next Block.
    block=Block(32,4,qkv_bias=True).eval()
    block.conv_neck=activated_neck
    with torch.inference_mode():
        block.conv_neck_enabled=False;plain=block(tokens,grid=(4,2))
        block.conv_neck_enabled=True;block.conv_neck_placement='pre_attn'
        pre=block(tokens,grid=(4,2))
        block.conv_neck_placement='post_mlp';post=block(tokens,grid=(4,2))
        assert not torch.allclose(pre[:,0],plain[:,0],atol=1e-7,rtol=0)
        assert torch.equal(post[:,0],plain[:,0])
        assert not torch.equal(post[:,1:],plain[:,1:])
    return dict(official_neck_equivalent=True,identity_initialization_exact=True,
        rectangular_grid_and_cls_bypass=True,pre_attn_changes_same_block_cls=True,
        post_mlp_cls_timing_verified=True,synthetic_neck_gradients_finite=True,
        zero_init_last_norm_receives_gradient=True,optimizer_steps=0)


def run(report_path):
    from config import cfg as original
    from mldc_vitmatte_cnl.model.make_model_mldc import MLDCReID as Previous,add_mldc_config as previous_config
    from .verify import real_batch,reset_seed
    reset_seed();torch.set_num_threads(1);started=time.monotonic()
    package=Path(__file__).resolve().parent
    configs={}
    for name in ('neck_all','neck_grouped','neck_all_pre_attn','dcm','baseline','dcm_mean'):
        cfg=add_mldc_config(original.clone())
        cfg.merge_from_file(str(package/'configs'/(name+'.yml')));cfg.freeze()
        assert cfg.SOLVER.SEED==1234
        assert '/mldc_vitmatte_neck_cnl/' in cfg.OUTPUT_DIR
        configs[name]=cfg
    assert configs['neck_all'].MODEL.CONV_NECK_LAYERS==list(range(1,13))
    assert configs['neck_all'].MODEL.CONV_NECK_PLACEMENT=='post_mlp'
    assert configs['neck_all_pre_attn'].MODEL.CONV_NECK_PLACEMENT=='pre_attn'
    assert configs['neck_grouped'].MODEL.CONV_NECK_PLACEMENT=='post_mlp'
    assert not configs['dcm'].MODEL.CONV_NECK_ENABLED
    assert not configs['baseline'].MODEL.DETAIL_ENABLED
    print('CODE_CHECK original convolution and CLS position semantics',flush=True)
    checks=verify_neck(package);gc.collect()
    cfg=configs['neck_all'];images,labels,cameras,classes=real_batch(cfg)
    # Sequential models keep this check within the server's 2GB memory limit.
    print('CODE_CHECK previous SSL/DCM initialization reference',flush=True)
    prior_cfg=previous_config(original.clone())
    prior_cfg.merge_from_file('mldc_vitmatte_cnl/configs/dcm.yml');prior_cfg.freeze()
    reset_seed();previous=Previous(prior_cfg,classes,8,1).eval()
    expected_digest=common_digest(previous)
    expected_rng=torch.random.get_rng_state().clone()
    with torch.inference_mode():expected=previous(images[:2]).clone()
    del previous;gc.collect()
    print('CODE_CHECK all 12 Necks on real 256x128 inputs',flush=True)
    reset_seed();model=MLDCReID(cfg,classes,8,1).eval()
    assert common_digest(model)==expected_digest
    assert torch.equal(torch.random.get_rng_state(),expected_rng)
    versions={k:p._version for k,p in model.named_parameters()}
    buffers={k:b.clone() for k,b in model.named_buffers()}
    visits={};hooks=[]
    for index,block in enumerate(model.base.blocks,1):
        assert block.conv_neck is not None
        assert block.conv_neck_placement=='post_mlp'
        def record(module,inputs,output,index=index):
            visits[index]=dict(shape=list(output.shape),calls=visits.get(index,{}).get('calls',0)+1)
        hooks.append(block.conv_neck.register_forward_hook(record))
    with torch.inference_mode():
        data=model(images[:2],return_details=True,return_maps=True)
        assert torch.equal(data['fused_unit'],expected)
        assert data['fused_unit'].shape==(2,1024)
        assert torch.isfinite(data['fused_unit']).all()
        assert torch.allclose(data['fused_unit'].norm(dim=1),torch.ones(2),atol=1e-6)
        assert torch.allclose(data['fused_unit']@data['fused_unit'].T,
            .9*(data['global_unit']@data['global_unit'].T)+.1*(data['detail_unit']@data['detail_unit'].T),atol=1e-6)
    assert all(v==dict(shape=[2,768,16,8],calls=1) for v in visits.values()) and len(visits)==12
    for hook in hooks:hook.remove()
    assert all(p._version==versions[k] and p.grad is None for k,p in model.named_parameters())
    assert all(torch.equal(b,buffers[k]) for k,b in model.named_buffers())
    del expected,buffers,versions;gc.collect()
    print('CODE_CHECK optimizer inclusion and strict checkpoint format (no steps)',flush=True)
    from solver import make_optimizer
    from .loss.make_loss_mldc import make_loss
    _,center=make_loss(cfg,classes)
    optimizer,_=make_optimizer(cfg,model,center)
    configure_head_learning_rates(cfg,model,optimizer)
    active={id(p) for p in model.parameters() if p.requires_grad}
    covered=[id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(covered)==len(set(covered)) and set(covered)==active
    rates={id(p):group['lr'] for group in optimizer.param_groups for p in group['params']}
    assert rates[id(model.base.blocks[0].conv_neck.conv1.weight)]==rates[id(model.base.cls_token)]
    del optimizer,center,rates,active,covered;gc.collect()
    architecture=model.architecture()
    neck_count=sum(p.numel() for name,p in model.named_parameters() if '.conv_neck.' in name)
    assert neck_count==23040000
    temporary=Path(report_path).with_suffix('.roundtrip.pth')
    assert not temporary.exists()
    try:
        from .processor.processor_mldc import save_checkpoint
        save_checkpoint(model,cfg,0,temporary)
        expected=data['fused_unit'].clone()
        del model,data;gc.collect()
        model=MLDCReID(cfg,classes,8,1,initialize=False).eval()
        model.load_param(temporary)
        assert model.checkpoint_epoch==0
        with torch.inference_mode():assert torch.equal(model(images[:2]),expected)
    finally:
        temporary.unlink(missing_ok=True)
    print('CODE_CHECK prior trained global weights import',flush=True)
    checkpoint=Path('/root/log/transreid/night600/mldc_learnable_global_cnl/deep_four_uniform_cross_camera_cnl_fixed_equal_schedule/transformer_120.pth')
    before=checkpoint.stat()
    model.load_global_checkpoint(checkpoint)
    with torch.inference_mode():
        with_necks=model(images[:2],return_details=True)['global_unit'].clone()
        for block in model.base.blocks:block.conv_neck_enabled=False
        without_necks=model(images[:2],return_details=True)['global_unit']
        assert torch.equal(with_necks,without_necks)
    after=checkpoint.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    del model;gc.collect()
    grouped=MLDCReID(configs['neck_grouped'],classes,8,1,initialize=False).eval()
    assert [i for i,b in enumerate(grouped.base.blocks,1) if b.conv_neck is not None]==[3,6,9,12]
    with torch.inference_mode():assert grouped(images[:1]).shape==(1,1024)
    del grouped;gc.collect()
    report=dict(seed=1234,device='cpu',all_passed=True,mode='forward_and_synthetic_gradient_checks',
        optimizer_steps=0,full_training_started=False,retrieval_performance_measured=False,
        original_checkpoint_unchanged=True,forward_parameters_unchanged=True,forward_buffers_unchanged=True,
        previous_dcm_initialization_exact=True,baseline_rng_preserved=True,initial_output_matches_previous_dcm=True,
        config_variants=list(configs),all12_neck_visits=visits,strict_checkpoint_roundtrip_exact=True,
        explicit_global_weight_import=True,identity_neck_disabled_equivalent=True,
        optimizer_coverage_exact=True,neck_learning_rate_matches_backbone=True,
        grouped_post_mlp_forward=True,descriptor_dim=1024,neck_added_parameters=neck_count,
        architecture=architecture,small_module_checks=checks,gpu_capacity_validated=False,
        seconds=time.monotonic()-started)
    Path(report_path).write_text(json.dumps(report,indent=2)+'\n')
    print('VERIFICATION_RESULT '+json.dumps(report),flush=True)
