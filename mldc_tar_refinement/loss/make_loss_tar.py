"""Separate MLDC, TAR CLS and final-residual ID + batch-hard supervision."""
import torch
import torch.nn.functional as F
from loss.triplet_loss import euclidean_dist, hard_example_mining


def make_loss(cfg, num_classes=None):
    if cfg.MODEL.IF_LABELSMOOTH != 'off' or cfg.MODEL.IF_WITH_CENTER != 'no':
        raise ValueError('Retain unsmoothed ID loss without center loss')

    def triplet(feature, target):
        positive, negative = hard_example_mining(euclidean_dist(feature.float(), feature.float()), target)
        if cfg.MODEL.NO_MARGIN:
            return F.softplus(positive - negative).mean()
        return F.relu(positive - negative + cfg.SOLVER.MARGIN).mean()

    def loss_fn(parts, target):
        with torch.autocast(device_type=target.device.type, enabled=False):
            terms = {}
            for name, raw in [('global', 'raw_readout'), ('tar', 'raw_tar'), ('residual', 'raw_residual')]:
                terms[name + '_id'] = F.cross_entropy(parts[name + '_logits'].float(), target)
                terms[name + '_triplet'] = triplet(parts[raw], target)
            terms['tar_kl'] = parts['tar_kl']
            terms['tar_kl_weighted'] = parts['tar_kl_weighted']
            def reid(name):
                return (cfg.MODEL.ID_LOSS_WEIGHT * terms[name + '_id'] +
                        cfg.MODEL.TRIPLET_LOSS_WEIGHT * terms[name + '_triplet'])
            total = (reid('global') + cfg.SOLVER.TAR_REID_WEIGHT * reid('tar') +
                     cfg.SOLVER.RESIDUAL_REID_WEIGHT * reid('residual') +
                     cfg.SOLVER.TAR_KL_WEIGHT * terms['tar_kl_weighted'])
            if not torch.isfinite(total).item():
                raise FloatingPointError('Non-finite MLDC + TAR objective')
            loss_fn.last_terms = {name: value.detach() for name, value in terms.items()}
            return total

    loss_fn.last_terms = {}
    return loss_fn
