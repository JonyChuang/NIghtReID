"""Original MLDC SGD and cosine factory, with independent heads at the same LR."""
import torch

HEADS=('bottleneck.','classifier.','last_bottleneck.','last_classifier.','concat_bottleneck.','concat_classifier.')


def build_optimizer(cfg,model):
    if cfg.SOLVER.OPTIMIZER_NAME!='SGD' or cfg.SOLVER.LARGE_FC_LR:
        raise ValueError('Use matched SGD and explicit head LR factor')
    groups=[]
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        head=name.startswith(HEADS);bias='bias' in name
        factor=(cfg.SOLVER.MLDC_HEAD_LR_FACTOR if head else 1.)*(cfg.SOLVER.BIAS_LR_FACTOR if bias else 1.)
        groups.append(dict(params=[p],lr=cfg.SOLVER.BASE_LR*factor,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY_BIAS if bias else cfg.SOLVER.WEIGHT_DECAY,
            category='head' if head else 'backbone',parameter_name=name))
    return torch.optim.SGD(groups,momentum=cfg.SOLVER.MOMENTUM)


def build_scheduler(cfg,optimizer):
    if cfg.SOLVER.WARMUP_METHOD!='cosine':raise ValueError('Use the original cosine warmup schedule')
    from solver.scheduler_factory import create_scheduler
    return create_scheduler(cfg,optimizer)


def lr_summary(optimizer):
    return {name:sorted({float(g['lr']) for g in optimizer.param_groups if g['category']==name}) for name in ('backbone','head')}
