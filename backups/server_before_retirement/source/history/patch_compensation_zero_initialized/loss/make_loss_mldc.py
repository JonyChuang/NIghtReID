"""The existing identity + triplet objective for a single descriptor."""
import torch
import torch.nn.functional as F
from loss.center_loss import CenterLoss
from loss.softmax_loss import CrossEntropyLabelSmooth
from loss.triplet_loss import euclidean_dist, hard_example_mining


def make_loss(cfg, num_classes):
    if cfg.DATALOADER.SAMPLER != 'softmax_triplet' or cfg.MODEL.METRIC_LOSS_TYPE != 'triplet' or cfg.MODEL.IF_WITH_CENTER != 'no':
        raise ValueError('MLDC experiment uses identity + triplet without center loss')
    smoothed = CrossEntropyLabelSmooth(num_classes=num_classes) if cfg.MODEL.IF_LABELSMOOTH == 'on' else None

    def loss_fn(score, feature, target, target_cam=None):
        with torch.autocast(device_type=feature.device.type, enabled=False):
            score, feature = score.float(), feature.float()
            identity = smoothed(score, target) if smoothed is not None else F.cross_entropy(score, target)
            positive, negative = hard_example_mining(euclidean_dist(feature, feature), target)
            gap = positive - negative
            triplet = F.softplus(gap).mean() if cfg.MODEL.NO_MARGIN else F.relu(gap + cfg.SOLVER.MARGIN).mean()
            total = cfg.MODEL.ID_LOSS_WEIGHT * identity + cfg.MODEL.TRIPLET_LOSS_WEIGHT * triplet
            if not torch.isfinite(total).item():
                raise FloatingPointError('Non-finite MLDC loss')
            loss_fn.last_terms = {'identity': identity.detach(), 'triplet': triplet.detach()}
            return total

    loss_fn.last_terms = {}
    # Preserve the repository optimizer API; this criterion is disabled.
    center = CenterLoss(num_classes=num_classes, feat_dim=768, use_gpu=False)
    return loss_fn, center
