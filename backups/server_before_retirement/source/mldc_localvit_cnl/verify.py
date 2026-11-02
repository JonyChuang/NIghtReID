"""Real NightReID forward/gradient/checkpoint checks, with zero optimizer updates."""
import argparse
import ast
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time
import torch
from torch import nn
from .model.make_model_mldc import MLDCReID,add_mldc_config,configure_head_learning_rates
from .model.localvit_official import LocalityFeedForward
from .model.backbones.vit_pytorch_mldc import Block


def seed():
    import numpy as np
    random.seed(1234);np.random.seed(1234);torch.manual_seed(1234)


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_digest(model,common_only=False):
    value=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if '.local_ffn.' in name:continue
        if common_only and any(name.startswith('base.blocks.'+str(i)+'.'+prefix)
            for i in range(4) for prefix in ('mlp.','norm2.')):continue
        value.update(name.encode());value.update(digest(tensor).encode())
    return value.hexdigest()


def check_official(package):
    source=ast.parse((package/'references/localvit.py.txt').read_text())
    copied=ast.parse((package/'model/localvit_official.py').read_text())
    names={'h_sigmoid','h_swish','ECALayer','SELayer','LocalityFeedForward'}
    originals=[n for n in source.body if isinstance(n,ast.ClassDef) and n.name in names]
    extracted=[n for n in copied.body if isinstance(n,ast.ClassDef) and n.name in names]
    assert [ast.dump(n) for n in originals]==[ast.dump(n) for n in extracted]
    namespace=dict(nn=nn,torch=torch,math=math)
    exec(compile(ast.Module(body=originals,type_ignores=[]),'<official LocalViT>','exec'),namespace)
    local=LocalityFeedForward(32,32,1,4,act='hs+se',reduction=32).eval()
    reference=namespace['LocalityFeedForward'](32,32,1,4,act='hs+se',reduction=32).eval()
    reference.load_state_dict(local.state_dict(),strict=True)
    inputs=torch.randn(2,32,4,2)
    with torch.inference_mode():assert torch.equal(local(inputs),reference(inputs))
    block=Block(32,4,qkv_bias=True).eval()
    block.local_ffn=local;del block.mlp;del block.norm2
    tokens=torch.randn(2,9,32)
    with torch.inference_mode():
        attention_output=tokens+block.drop_path(block.attn(block.norm1(tokens)))
        actual=block(tokens,grid=(4,2))
        assert torch.equal(actual[:,:1],attention_output[:,:1])
        expected=reference(attention_output[:,1:].transpose(1,2).reshape(2,32,4,2))
        assert torch.equal(actual[:,1:],expected.flatten(2).transpose(1,2))


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
    images,labels,cameras,indices=[],[],[],[]
    for pid in chosen:
        for cam in sorted(groups[pid])[:2]:
            choices=groups[pid][cam];index=choices[int(rng.randint(len(choices)))]
            with Image.open(dataset.train[index][0]) as image:
                pixels=np.asarray(image.convert('RGB').resize((128,256),Image.Resampling.BICUBIC)).copy()
            images.append(torch.from_numpy(pixels).permute(2,0,1).float()/255)
            labels.append(pid);cameras.append(cam);indices.append(index)
    image=torch.stack(images)
    image=(image-torch.tensor(cfg.INPUT.PIXEL_MEAN)[None,:,None,None])/torch.tensor(cfg.INPUT.PIXEL_STD)[None,:,None,None]
    return image,torch.tensor(labels),torch.tensor(cameras),torch.tensor(indices),dataset


def run(report_path):
    from config import cfg as original
    from mldc_learnable_global_cnl.model.make_model_mldc import MLDCReID as Original
    from .processor.processor_mldc import save_checkpoint
    started=time.monotonic();torch.set_num_threads(1);seed()
    package=Path(__file__).resolve().parent
    cfg=add_mldc_config(original.clone());cfg.merge_from_file(str(package/'configs/localvit_first4_nightreid.yml'));cfg.freeze()
    control=add_mldc_config(original.clone());control.merge_from_file(str(package/'configs/baseline_nightreid.yml'));control.freeze()
    assert cfg.DATASETS.NAMES=='nightreid' and cfg.SOLVER.SEED==1234
    assert cfg.SOLVER.CNL_GAMMA==.9 and cfg.SOLVER.CNL_WEIGHT==.1
    assert not Path(cfg.OUTPUT_DIR).exists(),'Verification must not create a training output directory'
    print('LOCALVIT_CHECK official module, patch residual, rectangular grid and CLS bypass',flush=True)
    check_official(package);gc.collect()
    images,labels,cameras,indices,dataset=real_batch(cfg)
    classes=dataset.num_train_pids
    print('LOCALVIT_CHECK original SSL baseline and disabled control',flush=True)
    seed();baseline=Original(cfg,classes,6,1).eval()
    old_all=state_digest(baseline);old_common=state_digest(baseline,True)
    old_rng=torch.random.get_rng_state().clone()
    old_count=sum(p.numel() for p in baseline.parameters())
    mapped={}
    for index in range(4):
        for linear in ('fc1','fc2'):
            mapped[(index,linear)]=digest(getattr(baseline.base.blocks[index].mlp,linear).weight)
    with torch.inference_mode():old_output=baseline(images[:2]).clone()
    del baseline;gc.collect()
    seed();disabled=MLDCReID(control,classes,6,1).eval()
    assert state_digest(disabled)==old_all
    with torch.inference_mode():assert torch.equal(disabled(images[:2]),old_output)
    del disabled,old_output;gc.collect()
    print('LOCALVIT_CHECK pretrained mappings and first4-only integration',flush=True)
    seed();model=MLDCReID(cfg,classes,6,1).eval()
    assert state_digest(model,True)==old_common
    assert torch.equal(torch.random.get_rng_state(),old_rng)
    for index,block in enumerate(model.base.blocks):
        if index<4:
            assert isinstance(block.local_ffn,LocalityFeedForward)
            assert not hasattr(block,'mlp') and not hasattr(block,'norm2')
            assert digest(block.local_ffn.conv[0].weight[:,:,0,0])==mapped[(index,'fc1')]
            assert digest(block.local_ffn.conv[-2].weight[:,:,0,0])==mapped[(index,'fc2')]
            dw=block.local_ffn.conv[3]
            assert dw.groups==dw.in_channels==dw.out_channels==3072
            assert dw.kernel_size==(3,3)
            assert block.local_ffn.conv[6].fc[0].out_features==4
        else:
            assert block.local_ffn is None and hasattr(block,'mlp') and hasattr(block,'norm2')
    visits={};hooks=[]
    for index in range(4):
        def record(module,inputs,output,index=index):visits[index+1]=list(output.shape)
        hooks.append(model.base.blocks[index].local_ffn.register_forward_hook(record))
    with torch.inference_mode():
        output,raw,layers=model(images[:2],return_components=True)
        assert output.shape==raw.shape==(2,768) and layers.shape==(12,2,768)
        assert torch.isfinite(output).all()
    for hook in hooks:hook.remove()
    assert visits=={i:[2,768,16,8] for i in range(1,5)}
    print('LOCALVIT_CHECK optimizer coverage, then real-image gradient path (zero steps)',flush=True)
    from solver import make_optimizer
    from .loss.make_loss_mldc import make_loss
    from .global_cnl import GlobalCNL
    loss_fn,center=make_loss(cfg,classes)
    optimizer,_=make_optimizer(cfg,model,center);configure_head_learning_rates(cfg,model,optimizer)
    ids=[id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(ids)==len(set(ids)) and set(ids)=={id(p) for p in model.parameters() if p.requires_grad}
    rates={id(p):group['lr'] for group in optimizer.param_groups for p in group['params']}
    assert rates[id(model.base.blocks[0].local_ffn.conv[3].weight)]==rates[id(model.base.cls_token)]
    del optimizer,center,ids,rates;gc.collect()
    # Freeze unchanged backbone weights in this temporary check only, reducing
    # gradient storage on the 2GB CPU server. Gradients still propagate through
    # every later attention/FFN to all four new LocalFFNs.
    flags={name:p.requires_grad for name,p in model.named_parameters()}
    for name,p in model.named_parameters():
        p.requires_grad_(flags[name] and ('.local_ffn.' in name or name.startswith(('mldc.','bottleneck.','classifier.'))))
    versions={name:p._version for name,p in model.named_parameters()}
    model.train()
    scores,raw,units=model(images,cam_label=cameras,return_cnl_features=True)
    memory=GlobalCNL(labels,cameras,768,gamma=cfg.SOLVER.CNL_GAMMA)
    cnl,stats=memory(units,torch.arange(4),labels,cameras)
    loss=loss_fn(scores,raw,labels,cameras)+cfg.SOLVER.CNL_WEIGHT*cnl
    assert torch.isfinite(loss)
    loss.backward()
    gradients={}
    for index in range(4):
        module=model.base.blocks[index].local_ffn
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        gradients[index+1]=float(module.conv[3].weight.grad.norm())
        assert gradients[index+1]>0
    assert all(p._version==versions[name] for name,p in model.named_parameters())
    assert not memory.valid.any() # Forward must not update the CNL memory.
    terms={k:float(v) for k,v in loss_fn.last_terms.items()};terms['cnl']=float(cnl.detach())
    model.zero_grad(set_to_none=True)
    for name,p in model.named_parameters():p.requires_grad_(flags[name])
    del scores,raw,units,loss,cnl,memory,loss_fn;gc.collect()
    model.eval()
    with torch.inference_mode():expected=model(images[:2]).clone()
    architecture=model.architecture();count=sum(p.numel() for p in model.parameters())
    print('LOCALVIT_CHECK strict checkpoint restore',flush=True)
    temporary=Path(report_path).with_suffix('.roundtrip.pth')
    assert not temporary.exists()
    try:
        save_checkpoint(model,cfg,0,temporary)
        del model;gc.collect()
        restored=MLDCReID(cfg,classes,6,1,initialize=False).eval()
        restored.load_param(temporary)
        with torch.inference_mode():assert torch.equal(restored(images[:2]),expected)
        assert restored.checkpoint_epoch==0
    finally:
        temporary.unlink(missing_ok=True)
    assert not Path(cfg.OUTPUT_DIR).exists()
    report=dict(all_passed=True,seed=1234,dataset='nightreid',device='cpu',optimizer_steps=0,
        full_training_started=False,training_output_created=False,retrieval_performance_measured=False,
        dataset_protocol=dict(train_ids=500,train_images=15514,query_images=1394,gallery_images=26111),
        official_classes_ast_identical=True,official_local_ffn_forward_exact=True,
        cls_bypasses_complete_local_ffn=True,local_ffn_visits=visits,first4_only=True,
        unchanged_parameter_initialization_exact=True,linear_to_conv_mappings_exact=True,
        caller_rng_preserved=True,disabled_baseline_output_exact=True,
        parameter_count=count,baseline_parameter_count=old_count,net_parameter_increase=count-old_count,
        optimizer_coverage_exact=True,local_dwconv_lr_matches_backbone=True,
        real_image_local_ffn_backward=True,dwconv_gradient_norms=gradients,
        gradient_scope='temporary freeze of unchanged backbone weights; LocalFFNs/MLDC/BN/classifier checked',
        model_parameters_not_updated=True,temporary_training_bn_buffers_updated=True,
        cnl_memory_not_updated=True,loss_terms=terms,cnl_stats=stats,
        strict_checkpoint_roundtrip_exact=True,architecture=architecture,
        gpu_batch64_validated=False,seconds=time.monotonic()-started)
    Path(report_path).write_text(json.dumps(report,indent=2)+'\n')
    print('VERIFICATION_RESULT '+json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',required=True)
    run(parser.parse_args().report)
