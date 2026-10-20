"""Global ID/Triplet plus local whole-descriptor ID and guided Triplet."""
import torch
import torch.nn.functional as F
from loss.triplet_loss import euclidean_dist, hard_example_mining


def local_triplet(local, global_feature, target, candidates=8, margin=0.3):
    distance = euclidean_dist(local.float(), local.float())
    same = target[:, None].eq(target[None, :])
    positive_mask = same & ~torch.eye(len(target), dtype=torch.bool, device=target.device)
    negative_mask = ~same
    if not positive_mask.any(1).all() or not negative_mask.any(1).all():
        raise ValueError('Triplet batch needs >=2 identities and >=2 instances per identity')
    hardest_positive = distance.masked_fill(~positive_mask, -torch.inf).max(1).values
    if candidates > 0:
        global_unit = F.normalize(global_feature.detach().float(), dim=1)
        similarities = global_unit @ global_unit.T
        count = min(int(candidates), int(negative_mask.sum(1).min()))
        indices = similarities.masked_fill(~negative_mask, -torch.inf).topk(count, dim=1).indices
        selected = torch.zeros_like(negative_mask).scatter_(1, indices, True)
    else:
        selected = negative_mask
    hardest_negative = distance.masked_fill(~selected, torch.inf).min(1).values
    return F.relu(hardest_positive-hardest_negative+margin).mean()


def make_loss(cfg, num_classes=None):
    if cfg.MODEL.IF_LABELSMOOTH != 'off' or cfg.MODEL.IF_WITH_CENTER != 'no':
        raise ValueError('This experiment retains the unsmoothed ID loss without center loss')

    def loss_fn(parts, target, phase):
        with torch.autocast(device_type=target.device.type, enabled=False):
            global_id = F.cross_entropy(parts['global_logits'].float(), target)
            positive, negative = hard_example_mining(
                euclidean_dist(parts['raw_readout'].float(), parts['raw_readout'].float()), target)
            global_tri = (F.softplus(positive-negative).mean() if cfg.MODEL.NO_MARGIN else
                          F.relu(positive-negative+cfg.SOLVER.MARGIN).mean())
            local_id = F.cross_entropy(parts['local_logits'].float(), target)
            local_tri = local_triplet(parts['raw_lpam'], parts['mldc_after_bn'], target,
                                      cfg.SOLVER.LPAM_NEG_CANDIDATES, cfg.SOLVER.LPAM_MARGIN)
            overlap = parts['overlap_loss'].float()
            total = cfg.SOLVER.LPAM_LOCAL_WEIGHT*(local_id+local_tri) + cfg.SOLVER.LPAM_DIVERSITY_WEIGHT*overlap
            if phase == 'joint':
                total = total + cfg.MODEL.ID_LOSS_WEIGHT*global_id + cfg.MODEL.TRIPLET_LOSS_WEIGHT*global_tri
            if not torch.isfinite(total).item():
                raise FloatingPointError('Non-finite LPAM v2 objective')
            loss_fn.last_terms = {name: value.detach() for name, value in
                                  [('global_id', global_id), ('global_triplet', global_tri),
                                   ('local_id', local_id), ('local_triplet', local_tri), ('overlap', overlap)]}
            return total

    loss_fn.last_terms = {}
    return loss_fn
