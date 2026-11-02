"""Match the fixed-equal CNL schedule and retain the MLDC logit LR multiplier."""
import math


def create_fixed_equal_scheduler(cfg, optimizer):
    """Use the same epoch factors as mldc_feature_enhancement.cppi_optim.

    Scale each group's base LR, preserving head and bias multipliers during
    warmup and at the cosine floor. Called once, before any training batches.
    """
    from torch.optim.lr_scheduler import LambdaLR

    warmup, total = cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.MAX_EPOCHS

    def factor(epoch):
        if epoch < warmup:
            return .01 + .99 * epoch / warmup
        if epoch >= total:
            return .002
        return .002 + .998 * .5 * (1 + math.cos(math.pi * epoch / total))

    return LambdaLR(optimizer, factor)


class WeightLRScheduler:
    def __init__(self, scheduler, optimizer, groups, factor):
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.groups = tuple(groups)
        self.factor = float(factor)
        self._scale()

    def _scale(self):
        for index in self.groups:
            self.optimizer.param_groups[index]['lr'] *= self.factor

    def step(self, *args, **kwargs):
        result = self.scheduler.step(*args, **kwargs)
        self._scale()
        return result

    def state_dict(self):
        return {'scheduler': self.scheduler.state_dict(), 'groups': self.groups, 'factor': self.factor}

    def load_state_dict(self, state):
        if tuple(state['groups']) != self.groups or state['factor'] != self.factor:
            raise ValueError('MLDC scheduler configuration differs from saved state')
        self.scheduler.load_state_dict(state['scheduler'])


def wrap_weight_scheduler(cfg, model, optimizer, scheduler):
    factor = float(getattr(cfg.SOLVER, 'MLDC_WEIGHT_LR_FACTOR', 1.0))
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError('MLDC weight learning-rate factor must be finite and positive')
    if factor == 1.0:
        return scheduler
    parameter = model.mldc.layer_weights
    if not parameter.requires_grad:
        raise ValueError('MLDC weight LR factor requires a learnable MLDC readout')
    groups = []
    for index, group in enumerate(optimizer.param_groups):
        if any(p is parameter for p in group['params']):
            if len(group['params']) != 1:
                raise ValueError('MLDC weights must have an isolated optimizer group')
            groups.append(index)
    if len(groups) != 1:
        raise ValueError('Expected exactly one MLDC weight optimizer group')
    return WeightLRScheduler(scheduler, optimizer, groups, factor)
