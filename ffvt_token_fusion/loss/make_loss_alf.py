"""Independent global/fusion ID and triplet supervision, evaluated in FP32."""
import torch
import torch.nn.functional as F

from loss.center_loss import CenterLoss
from loss.softmax_loss import CrossEntropyLabelSmooth
from loss.triplet_loss import euclidean_dist, hard_example_mining


def make_alf_loss(cfg, num_classes):
    if cfg.DATALOADER.SAMPLER != 'softmax_triplet' or cfg.MODEL.METRIC_LOSS_TYPE != 'triplet':
        raise ValueError('FFVT requires the Night600 softmax_triplet sampler and triplet loss')
    if cfg.MODEL.IF_WITH_CENTER != 'no':
        raise ValueError('FFVT experiment uses identity and triplet losses without center loss')
    smoothed_ce = (CrossEntropyLabelSmooth(num_classes=num_classes)
                   if cfg.MODEL.IF_LABELSMOOTH == 'on' else None)

    def identity_loss(score, target):
        return (smoothed_ce(score, target) if smoothed_ce is not None
                else F.cross_entropy(score, target))

    def triplet_loss(feat, target):
        distances = euclidean_dist(feat, feat)
        positive, negative = hard_example_mining(distances, target)
        gap = positive - negative
        return (F.softplus(gap).mean() if cfg.MODEL.NO_MARGIN
                else F.relu(gap + cfg.SOLVER.MARGIN).mean())

    def loss_func(score, feat, target, target_cam):
        # The ViT forward uses mixed precision; distance squares, mining and
        # soft-margin evaluation remain FP32 to avoid FP16 overflow.
        with torch.autocast(device_type=feat['cls'].device.type, enabled=False):
            global_feature = feat['cls'].float()
            parts = feat['parts'].float()
            part_scores = score['parts']
            num_parts = parts.shape[1]
            if len(part_scores) != num_parts or num_parts < 1:
                raise ValueError('Every fusion branch requires its own identity classifier')
            cls_identity = identity_loss(score['cls'].float(), target)
            parts_identity = torch.stack([
                identity_loss(logits.float(), target) for logits in part_scores]).mean()
            # Average all global/part classifiers, rather than letting the
            # number of queries increase the overall identity-loss weight.
            identity = (cls_identity + num_parts * parts_identity) / (num_parts + 1)
            cls_triplet = triplet_loss(global_feature, target)
            parts_triplet = triplet_loss(parts.flatten(1), target)
            metric = 0.5 * (cls_triplet + parts_triplet)
            total = (cfg.MODEL.ID_LOSS_WEIGHT * identity
                     + cfg.MODEL.TRIPLET_LOSS_WEIGHT * metric)
            if not torch.isfinite(total).item():
                terms = {'cls_id': cls_identity, 'parts_id': parts_identity,
                         'cls_tri': cls_triplet, 'parts_tri': parts_triplet}
                invalid = [name for name, value in terms.items()
                           if not torch.isfinite(value).item()]
                raise FloatingPointError('Non-finite FFVT loss: terms={}, CLS finite={}, parts finite={}'.format(
                    invalid, torch.isfinite(global_feature).all().item(),
                    torch.isfinite(parts).all().item()))
            loss_func.last_terms = {
                'cls_id': cls_identity.detach(), 'parts_id': parts_identity.detach(),
                'cls_tri': cls_triplet.detach(), 'parts_tri': parts_triplet.detach(),
            }
            return total

    loss_func.last_terms = {}
    # Retain the project's optimizer signature; center loss is disabled above.
    center = CenterLoss(num_classes=num_classes, feat_dim=2048,
                        use_gpu=torch.cuda.is_available())
    return loss_func, center
