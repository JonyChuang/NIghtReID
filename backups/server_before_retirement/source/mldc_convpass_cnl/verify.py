"""Verify retained baseline function and Convpass gradients without optimizer updates."""
import argparse
import ast
import gc
import hashlib
import json
from pathlib import Path
import random
import time
import torch
from torch import nn
from .model.make_model_mldc import MLDCReID,add_mldc_config,configure_head_learning_rates
from .model.convpass import RectangularConvpass
from .model.convpass_official import Convpass


def seed():
    import numpy as np
    random.seed(1234);np.random.seed(1234);torch.manual_seed(1234)


def is_adapter(name):
    return '.adapter_attn.' in name or '.adapter_mlp.' in name


def state_digest(model,backbone_only=False):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if is_adapter(name) or (backbone_only and not name.startswith('base.')):continue
        digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def check_official(package):
    original=ast.parse((package/'references/convpass.py.txt').read_text())
    copied=ast.parse((package/'model/convpass_official.py').read_text())
    names={'QuickGELU','Convpass'}
    a=[ast.dump(n) for n in original.body if isinstance(n,ast.ClassDef) and n.name in names]
    b=[ast.dump(n) for n in copied.body if isinstance(n,ast.ClassDef) and n.name in names]
    assert a==b
    adapter=RectangularConvpass().eval();reference=Convpass().eval()
    # Use a nonzero, temporary test fixture so equivalence is not vacuously
    # established by the all-zero output projection.
    with torch.no_grad():
        adapter.adapter_up.weight.normal_(std=.02)
        adapter.adapter_conv.weight.normal_(std=.02)
    reference.load_state_dict(adapter.state_dict(),strict=True)
    square=torch.randn(2,197,768);rectangle=torch.randn(2,129,768)
    with torch.inference_mode():
        assert torch.equal(adapter(square,(14,14)),reference(square))
        assert adapter(rectangle,(16,8)).shape==rectangle.shape
    adapter(rectangle,(16,8)).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in adapter.parameters())
    fresh=RectangularConvpass().eval()
    output=fresh(rectangle,(16,8))
    assert torch.count_nonzero(output)==0
    (output*torch.randn_like(output)).sum().backward()
    assert fresh.adapter_up.weight.grad.norm()>0
    assert fresh.adapter_conv.weight.grad.norm()==0 and fresh.adapter_down.weight.grad.norm()==0


def real_batch(cfg):
    import numpy as np
    from PIL import Image
    from datasets.nightreid import NightReID
    dataset=NightReID(root=cfg.DATASETS.ROOT_DIR,verbose=False)
    assert dataset.num_train_pids==500 and len(dataset.train)==15514
    assert len(dataset.query)==1394 and len(dataset.gallery)==26111
    groups={}
    for index,(_,pid,cam,_) in enumerate(dataset.train):
        groups.setdefault(pid,{}).setdefault(cam,[]).append(index)
    eligible=sorted(pid for pid,cams in groups.items() if len(cams)>=2)
    rng=np.random.RandomState(1234)
    chosen=[eligible[i] for i in rng.permutation(len(eligible))[:2]]
    images,labels,cameras=[],[],[]
    for pid in chosen:
        for cam in sorted(groups[pid])[:2]:
            choices=groups[pid][cam];index=choices[int(rng.randint(len(choices)))]
            with Image.open(dataset.train[index][0]) as image:
                pixels=np.asarray(image.convert('RGB').resize((128,256),Image.Resampling.BICUBIC)).copy()
            images.append(torch.from_numpy(pixels).permute(2,0,1).float()/255)
            labels.append(pid);cameras.append(cam)
    images=torch.stack(images)
    images=(images-torch.tensor(cfg.INPUT.PIXEL_MEAN)[None,:,None,None])/torch.tensor(cfg.INPUT.PIXEL_STD)[None,:,None,None]
    return images,torch.tensor(labels),torch.tensor(cameras),dataset


def run(report_path):
    from config import cfg as original
    from mldc_learnable_global_cnl.model.make_model_mldc import MLDCReID as Original
    from .processor.processor_mldc import save_checkpoint
    started=time.monotonic();torch.set_num_threads(1);seed()
    package=Path(__file__).resolve().parent
    configs={}
    for name in ('convpass_first4_frozen_nightreid','convpass_first4_joint_nightreid',
                 'convpass_all_frozen_nightreid','baseline_nightreid'):
        cfg=add_mldc_config(original.clone());cfg.merge_from_file(str(package/'configs'/(name+'.yml')));cfg.freeze()
        assert cfg.DATASETS.NAMES=='nightreid' and cfg.SOLVER.SEED==1234
        configs[name]=cfg
    cfg=configs['convpass_first4_frozen_nightreid']
    assert not Path(cfg.OUTPUT_DIR).exists()
    print('CONVPASS_CHECK official nonzero calculation and zero-up gradient bootstrap',flush=True)
    check_official(package);gc.collect()
    images,labels,cameras,dataset=real_batch(cfg);classes=dataset.num_train_pids
    print('CONVPASS_CHECK original SSL reference',flush=True)
    seed();baseline=Original(cfg,classes,6,1).eval()
    original_digest=state_digest(baseline);original_rng=torch.random.get_rng_state().clone()
    original_count=sum(p.numel() for p in baseline.parameters())
    with torch.inference_mode():expected,expected_raw,expected_layers=[v.clone() for v in baseline(images[:2],return_components=True)]
    del baseline;gc.collect()
    print('CONVPASS_CHECK retained weights and all-layer initial features',flush=True)
    seed();model=MLDCReID(cfg,classes,6,1).eval()
    assert state_digest(model)==original_digest
    assert torch.equal(torch.random.get_rng_state(),original_rng)
    visits={};hooks=[]
    for i,block in enumerate(model.base.blocks,1):
        assert hasattr(block,'mlp') and hasattr(block,'norm2')
        if i<=4:
            for site in ('adapter_attn','adapter_mlp'):
                module=getattr(block,site)
                def record(module,inputs,output,key=str(i)+'/'+site):
                    visits[key]=dict(shape=list(output.shape),max_absolute=float(output.abs().max()))
                hooks.append(module.register_forward_hook(record))
        else:assert block.adapter_attn is None and block.adapter_mlp is None
    with torch.inference_mode():
        output,raw,layers=model(images[:2],return_components=True)
        assert torch.equal(output,expected) and torch.equal(raw,expected_raw) and torch.equal(layers,expected_layers)
        max_difference=float((output-expected).abs().max())
    for hook in hooks:hook.remove()
    assert len(visits)==8 and all(v==dict(shape=[2,129,768],max_absolute=0.) for v in visits.values())
    trainable=[name for name,p in model.named_parameters() if p.requires_grad]
    assert all(is_adapter(n) or not n.startswith('base.') for n in trainable)
    adapter_count=sum(p.numel() for n,p in model.named_parameters() if is_adapter(n))
    total_count=sum(p.numel() for p in model.parameters())
    trainable_count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert adapter_count==109184 and total_count-original_count==adapter_count
    print('CONVPASS_CHECK optimizer inclusion and real-image frozen-backbone backward, zero updates',flush=True)
    from solver import make_optimizer
    from .loss.make_loss_mldc import make_loss
    from .global_cnl import GlobalCNL
    loss_fn,center=make_loss(cfg,classes)
    optimizer,_=make_optimizer(cfg,model,center);configure_head_learning_rates(cfg,model,optimizer)
    ids=[id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad}
    del optimizer,center,ids;gc.collect()
    backbone_before=state_digest(model,True)
    versions={name:p._version for name,p in model.named_parameters()}
    model.train()
    assert all(not module.training for module in model.base.modules() if isinstance(module,nn.modules.batchnorm._BatchNorm))
    scores,raw,units=model(images,cam_label=cameras,return_cnl_features=True)
    memory=GlobalCNL(labels,cameras,768,gamma=cfg.SOLVER.CNL_GAMMA)
    cnl,stats=memory(units,torch.arange(4),labels,cameras)
    loss=loss_fn(scores,raw,labels,cameras)+cfg.SOLVER.CNL_WEIGHT*cnl
    assert torch.isfinite(loss);loss.backward()
    gradients={}
    for i in range(4):
        for site in ('adapter_attn','adapter_mlp'):
            module=getattr(model.base.blocks[i],site)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
            value=float(module.adapter_up.weight.grad.norm());assert value>0
            assert module.adapter_down.weight.grad.norm()==0 and module.adapter_conv.weight.grad.norm()==0
            gradients[str(i+1)+'/'+site]=value
    assert all(p.grad is None for name,p in model.base.named_parameters() if not is_adapter('base.'+name))
    assert all(p._version==versions[name] for name,p in model.named_parameters())
    assert state_digest(model,True)==backbone_before and not memory.valid.any()
    terms={k:float(v) for k,v in loss_fn.last_terms.items()};terms['cnl']=float(cnl.detach())
    model.zero_grad(set_to_none=True)
    del scores,raw,units,loss,cnl,memory,loss_fn;gc.collect()
    model.eval();architecture=model.architecture()
    with torch.inference_mode():roundtrip_expected=model(images[:2]).clone()
    print('CONVPASS_CHECK strict checkpoint restore',flush=True)
    temporary=Path(report_path).with_suffix('.roundtrip.pth');assert not temporary.exists()
    try:
        save_checkpoint(model,cfg,0,temporary)
        del model;gc.collect()
        model=MLDCReID(cfg,classes,6,1,initialize=False).eval();model.load_param(temporary)
        with torch.inference_mode():assert torch.equal(model(images[:2]),roundtrip_expected)
    finally:temporary.unlink(missing_ok=True)
    # Joint control retains the exact same shape/initial function while allowing
    # the original active backbone parameters to be optimized.
    del model;gc.collect()
    seed();joint=MLDCReID(configs['convpass_first4_joint_nightreid'],classes,6,1).eval()
    assert joint.base.blocks[0].mlp.fc1.weight.requires_grad and joint.base.cls_token.requires_grad
    with torch.inference_mode():assert torch.equal(joint(images[:2]),expected)
    del joint;gc.collect()
    seed();all_layers=MLDCReID(configs['convpass_all_frozen_nightreid'],classes,6,1).eval()
    assert sum(b.adapter_attn is not None for b in all_layers.base.blocks)==12
    with torch.inference_mode():assert torch.equal(all_layers(images[:2]),expected)
    del all_layers;gc.collect()
    print('CONVPASS_CHECK trained baseline insertion equivalence',flush=True)
    checkpoint=Path('/root/log/transreid/nightreid/mldc_learnable_global_cnl/deep_four_uniform_cross_camera_cnl_fixed_equal_schedule/transformer_best.pth')
    before=checkpoint.stat()
    adapted=MLDCReID(cfg,classes,6,1,initialize=False).eval();adapted.load_global_checkpoint(checkpoint)
    with torch.inference_mode():trained_adapted=adapted(images[:2]).clone()
    del adapted;gc.collect()
    baseline=Original(cfg,classes,6,1).eval()
    state=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)['state_dict']
    state={(k[7:] if k.startswith('module.') else k):v for k,v in state.items()}
    baseline.load_state_dict(state,strict=True,assign=True)
    with torch.inference_mode():assert torch.equal(baseline(images[:2]),trained_adapted)
    after=checkpoint.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    assert not Path(cfg.OUTPUT_DIR).exists()
    report=dict(all_passed=True,seed=1234,dataset='nightreid',device='cpu',optimizer_steps=0,
        full_training_started=False,training_output_created=False,retrieval_performance_measured=False,
        official_classes_ast_identical=True,official_nonzero_forward_exact=True,
        original_parameter_initialization_exact=True,caller_rng_preserved=True,
        initial_all12_cls_and_descriptor_exact=True,initial_max_absolute_difference=max_difference,
        bypass_outputs=visits,original_ln_mlp_biases_retained=True,
        adapter_parameters=adapter_count,total_parameters=total_count,frozen_trainable_parameters=trainable_count,
        optimizer_coverage_exact=True,frozen_backbone_no_grad=True,frozen_backbone_bn_buffers_unchanged=True,
        up_projection_gradients=gradients,down_conv_zero_first_gradient_expected=True,
        activated_fixture_all_gradients_nonzero=True,model_parameters_not_updated=True,cnl_memory_not_updated=True,
        loss_terms=terms,cnl_stats=stats,strict_checkpoint_roundtrip_exact=True,
        joint_initial_output_exact=True,all12_initial_output_exact=True,
        trained_baseline_insertion_exact=True,original_checkpoint_unchanged=True,
        config_variants=list(configs),architecture=architecture,
        dataset_protocol=dict(train_ids=500,train_images=15514,query_images=1394,gallery_images=26111),
        gpu_batch64_validated=False,seconds=time.monotonic()-started)
    Path(report_path).write_text(json.dumps(report,indent=2)+'\n')
    print('VERIFICATION_RESULT '+json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',required=True)
    run(parser.parse_args().report)
