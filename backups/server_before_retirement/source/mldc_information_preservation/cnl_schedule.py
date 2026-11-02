"""Independent CNL loss warmup; the optimizer schedule and memory stay separate."""
import math


def auxiliary_weight_for_epoch(cfg, epoch):
    if cfg.SOLVER.AUXILIARY_LOSS != 'shared_cnl':
        return float(cfg.SOLVER.PRESERVE_WEIGHT)
    warmup = getattr(cfg.SOLVER, 'CNL_WARMUP_EPOCHS', 0)
    weight = float(cfg.SOLVER.CNL_WEIGHT)
    if type(warmup) is not int or warmup < 0:
        raise ValueError('CNL_WARMUP_EPOCHS must be a nonnegative integer')
    if type(epoch) is not int or epoch < 0:
        raise ValueError('Epoch must be a nonnegative integer')
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('CNL_WEIGHT must be finite and nonnegative')
    return weight if epoch > warmup else 0.


def auxiliary_loss_for_step(memory, weight, unit, layers, indices, metadata):
    # No neighbor mining or auxiliary gradient graph during warmup. The caller
    # still updates the memory after each successful original-loss step.
    if memory is None or weight == 0:
        return unit.new_zeros(()).float(), {}
    return memory(unit, layers, indices, metadata)
