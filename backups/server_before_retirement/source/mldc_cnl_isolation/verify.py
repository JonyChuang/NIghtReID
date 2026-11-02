"""Verify that paired no-fusion experiments differ only in CNL, without training."""
import argparse
import gc
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',required=True);args=parser.parse_args()
    import torch
    from torch.nn import functional as F
    from config import cfg as base_cfg
    from mldc_learnable_global_cnl.model.make_model_mldc import add_mldc_config,MLDCReID,configure_head_learning_rates
    from mldc_learnable_global_cnl.loss.make_loss_mldc import make_loss
    from mldc_learnable_global_cnl.global_cnl import GlobalCNL
    from mldc_learnable_global_cnl.scheduler_mldc import create_fixed_equal_scheduler
    from mldc_learnable_global_cnl.verify_schedule import main as verify_schedule
    from solver import make_optimizer
    from . import train
    from .processor import save_checkpoint

    torch.set_num_threads(1)
    configs=[]
    for filename in ['last_cls_no_cnl.yml','last_cls_cnl01.yml']:
        cfg=add_mldc_config(base_cfg.clone());cfg.merge_from_file(str(Path(__file__).parent/'configs'/filename));cfg.freeze();configs.append(cfg)
    def flat(node,prefix=''):
        result={}
        for k,v in node.items():
            if hasattr(v,'items'):result.update(flat(v,prefix+k+'.'))
            else:result[prefix+k]=v
        return result
    a,b=map(flat,configs)
    pair_differences={k:dict(no_cnl=a.get(k),cnl=b.get(k)) for k in a if a.get(k)!=b.get(k)}
    assert set(pair_differences)=={'SOLVER.CNL_WEIGHT','OUTPUT_DIR'}
    full=add_mldc_config(base_cfg.clone());full.merge_from_file('mldc_learnable_global_cnl/variants/deep_four_uniform_cross_camera_cnl/config.yml')
    c=flat(full)
    removal_differences={k:dict(full=c.get(k),last_cls=b.get(k)) for k in b if b.get(k)!=c.get(k)}
    assert set(removal_differences)=={'MODEL.MLDC_READOUT','OUTPUT_DIR'}
    def digest(model):
        result=hashlib.sha256()
        for name,tensor in model.state_dict().items():
            result.update(name.encode());result.update(tensor.cpu().numpy().tobytes())
        return result.hexdigest()
    torch.manual_seed(1234);control=MLDCReID(configs[0],300,8,1).eval();initial=digest(control)
    images=torch.randn(4,3,256,128,generator=torch.Generator().manual_seed(58))
    with torch.no_grad():expected=control(images).clone()
    del control;gc.collect()
    torch.manual_seed(1234);model=MLDCReID(configs[1],300,8,1).eval()
    assert digest(model)==initial
    with torch.no_grad():
        output,raw,layers=model(images,return_components=True)
        assert torch.equal(output,expected)
        assert torch.equal(raw,layers[-1])
        assert torch.equal(output,model.bottleneck(layers[-1]))
    assert model.readout=='last_cls' and not model.mldc.layer_weights.requires_grad
    assert model.readout_weights().tolist()==[0.]*11+[1.]
    assert not model.base.norm.weight.requires_grad
    model.train();labels=torch.tensor([0,0,1,1]);cameras=torch.tensor([0,1,0,1]);indices=torch.arange(4)
    scores,raw,unit=model(images,return_cnl_features=True)
    torch.testing.assert_close(unit,F.normalize(model.bottleneck(raw).float(),dim=1))
    loss_fn,center=make_loss(configs[1],300);base=loss_fn(scores,raw,labels,cameras)
    bank=GlobalCNL(labels.repeat(2),torch.cat((cameras,1-cameras)))
    # A nearby detached negative avoids a saturated zero loss on random images.
    direction=unit[0].detach().roll(1)
    keys=F.normalize(torch.stack([unit[0].detach()+.04*direction,unit[0].detach()-.04*direction]),dim=1)
    slots=torch.tensor([4,6])
    bank.update(keys,slots,bank.labels[slots],bank.cameras[slots])
    cnl,stats=bank(unit,indices,labels,cameras)
    assert cnl.item()>0 and stats['active_anchors']>0
    gradient=torch.autograd.grad(cnl,model.base.patch_embed.proj.weight,retain_graph=True)[0]
    assert torch.isfinite(gradient).all() and gradient.norm()>0
    total=base+configs[1].SOLVER.CNL_WEIGHT*cnl;total.backward()
    assert model.mldc.layer_weights.grad is None
    assert model.classifier.weight.grad is not None
    optimizer,_=make_optimizer(configs[1],model,center)
    configure_head_learning_rates(configs[1],model,optimizer)
    scheduler=create_fixed_equal_scheduler(configs[1],optimizer)
    assert not any(any(p is model.mldc.layer_weights for p in g['params']) for g in optimizer.param_groups)
    schedule=verify_schedule()
    report=dict(all_passed=True,optimizer_steps=0,training_started=False,
        paired_configuration_differences=pair_differences,removal_from_full_differences=removal_differences,
        paired_initial_state_identical=True,paired_initial_descriptor_identical=True,
        no_active_multilayer_fusion=True,readout='last transformer block CLS before final LayerNorm',
        retrieval_and_cnl='BN-after; L2 normalization for CNL and retrieval distance',
        cnl_loss=float(cnl.detach()),cnl_only_patch_projection_gradient_norm=float(gradient.norm()),
        cnl_stats=stats,full_aligned_schedule_verified=schedule['all_passed'],schedule_positions=121)
    Path(args.report).write_text(json.dumps(report,indent=2))
    print('VERIFICATION_RESULT '+json.dumps(report),flush=True)


if __name__=='__main__':main()
