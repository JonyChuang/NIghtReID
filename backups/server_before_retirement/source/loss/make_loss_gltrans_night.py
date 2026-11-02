"""GLTrans branch supervision from Eq. 16, using the host project's losses."""
import torch
import torch.nn.functional as F

from loss.center_loss import CenterLoss
from loss.softmax_loss import CrossEntropyLabelSmooth
from loss.triplet_loss import euclidean_dist, hard_example_mining


def make_gltrans_loss(cfg, num_classes):
    if 'triplet' not in cfg.DATALOADER.SAMPLER or cfg.MODEL.METRIC_LOSS_TYPE != 'triplet':
        raise ValueError('GLTrans Night requires identity sampling and triplet loss')
    smooth_ce = (CrossEntropyLabelSmooth(num_classes=num_classes)
                 if cfg.MODEL.IF_LABELSMOOTH == 'on' else None)

    def cross_entropy(logits, target):
        return smooth_ce(logits, target) if smooth_ce is not None else F.cross_entropy(logits, target)

    def stable_triplet(feature, target):
        # Same batch-hard distances and soft-margin objective as TripletLoss.
        # nn.SoftMarginLoss computes log(1 + exp(gap)) directly and overflows
        # around gap=88 in FP32; softplus is the stable equivalent.
        distance = euclidean_dist(feature, feature)
        dist_ap, dist_an = hard_example_mining(distance, target)
        gap = dist_ap - dist_an
        if cfg.MODEL.NO_MARGIN:
            return F.softplus(gap).mean()
        return F.relu(gap + cfg.SOLVER.MARGIN).mean()

    def loss_func(scores, features, target, target_cam):
        # scores: final CLS, GAE global, PTL local, CLS10, CLS11.
        # features: final CLS, GAE global, PTL local.
        if len(scores) != 5 or len(features) != 3:
            raise ValueError('GLTrans loss needs three main branches and two auxiliary CLS scores')
        # The caller uses CUDA autocast for the model forward. The project's
        # triplet distance squares and sums high-dimensional features, then
        # takes a square root. FP16 can overflow or produce non-finite
        # gradients at zero distance, so compute every loss term in FP32.
        with torch.autocast(device_type=features[0].device.type, enabled=False):
            ce_losses = [cross_entropy(score.float(), target) for score in scores]
            triplet_losses = [stable_triplet(feature.float(), target)
                              for feature in features]
            main_ce = sum(ce_losses[:3]) / 3
            main_triplet = sum(triplet_losses) / 3
            auxiliary_ce = sum(ce_losses[3:]) / 2
            total = (cfg.MODEL.ID_LOSS_WEIGHT * (main_ce + auxiliary_ce)
                     + cfg.MODEL.TRIPLET_LOSS_WEIGHT * main_triplet)
            if not torch.isfinite(total).item():
                names = ['CLS12 CE', 'GAE CE', 'PTL CE', 'CLS10 CE', 'CLS11 CE',
                         'CLS12 triplet', 'GAE triplet', 'PTL triplet']
                terms = ce_losses + triplet_losses
                bad_terms = [name for name, value in zip(names, terms)
                             if not torch.isfinite(value).item()]
                bad_features = [name for name, value in
                                zip(('CLS12', 'GAE', 'PTL'), features)
                                if not torch.isfinite(value).all().item()]
                raise FloatingPointError(
                    'Non-finite GLTrans loss; terms={}, features={}'.format(
                        bad_terms, bad_features))
            return total

    # The existing optimizer interface expects a center criterion even when the
    # configured loss is triplet-only. It is never applied to the GLTrans branches.
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=2048,
                                  use_gpu=torch.cuda.is_available())
    return loss_func, center_criterion
