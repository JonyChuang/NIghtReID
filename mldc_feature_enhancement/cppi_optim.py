"""Explicit parameter groups and a shared scalar schedule preserve LR ratios."""
import math
import torch


def build_optimizer(cfg, model):
    if cfg.SOLVER.OPTIMIZER_NAME != 'SGD' or cfg.SOLVER.WARMUP_METHOD != 'cosine':
        raise ValueError('CPPI uses the matched SGD/cosine protocol')
    groups = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        category = ('reader' if name.startswith(('cnn_reader.block.','cnn_reader.norm.')) else
                    'cppi' if name.startswith(('cppi_', 'cnn_reader.')) else
                    'head' if name.startswith(('bottleneck.', 'classifier.','shallow_bottleneck.','shallow_classifier.')) else 'backbone')
        is_sd=getattr(model,'mode',None)=='cnn_shallow_deep'
        multiplier = (float(cfg.SOLVER.CNN_SD_READER_LR_FACTOR) if category == 'reader' else
                      float(cfg.SOLVER.CNN_SD_LR_FACTOR if is_sd else cfg.SOLVER.CPPI_LR_FACTOR) if category == 'cppi' else
                      float(cfg.SOLVER.MLDC_HEAD_LR_FACTOR) if category == 'head' else 1.)
        bias = 'bias' in name
        if bias:
            multiplier *= cfg.SOLVER.BIAS_LR_FACTOR
        groups.append(dict(params=[parameter], lr=cfg.SOLVER.BASE_LR * multiplier,
            weight_decay=0. if name.endswith('.budget_logit') else (cfg.SOLVER.WEIGHT_DECAY_BIAS if bias else cfg.SOLVER.WEIGHT_DECAY),
            category=category, parameter_name=name, lr_multiplier=multiplier))
    return torch.optim.SGD(groups, momentum=cfg.SOLVER.MOMENTUM)


def build_scheduler(cfg, optimizer):
    warmup, total = cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.MAX_EPOCHS
    def factor(epoch):
        if epoch < warmup:
            return .01 + .99 * epoch / warmup
        if epoch >= total:
            return .002
        return .002 + .998 * .5 * (1 + math.cos(math.pi * epoch / total))
    # Unlike the original scalar warmup_lr_init/lr_min, this scales every
    # group's warmup and floor as well as its peak. Base trajectory is unchanged.
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def lr_summary(optimizer):
    return {category: sorted({float(g['lr']) for g in optimizer.param_groups if g['category'] == category})
            for category in ('backbone', 'cppi', 'reader', 'head')}
