"""Global baseline losses, local/part supervision, and the deployed fusion metric."""
import torch
import torch.nn.functional as F
from loss.triplet_loss import euclidean_dist, hard_example_mining


def mining_masks(target, global_feature, candidates=8, camera=None):
    same = target[:, None].eq(target[None, :])
    positive = same & ~torch.eye(len(target), dtype=torch.bool, device=target.device)
    negative = ~same
    if not positive.any(1).all() or not negative.any(1).all():
        raise ValueError('Triplet batch needs >=2 identities and >=2 instances per identity')
    if camera is not None:
        cross_camera = positive & camera[:, None].ne(camera[None, :])
        positive = torch.where(cross_camera.any(1, keepdim=True), cross_camera, positive)
    if candidates > 0:
        global_unit = F.normalize(global_feature.detach().float(), dim=1)
        similarities = global_unit @ global_unit.T
        count = min(int(candidates), int(negative.sum(1).min()))
        indices = similarities.masked_fill(~negative, -torch.inf).topk(count, dim=1).indices
        negative = torch.zeros_like(negative).scatter_(1, indices, True)
    return positive, negative


def masked_triplet(feature, positive, negative, margin):
    distance = euclidean_dist(feature.float(), feature.float())
    hardest_positive = distance.masked_fill(~positive, -torch.inf).max(1).values
    hardest_negative = distance.masked_fill(~negative, torch.inf).min(1).values
    return F.relu(hardest_positive-hardest_negative+margin).mean()


def make_loss(cfg, num_classes=None):
    if cfg.MODEL.IF_LABELSMOOTH != 'off' or cfg.MODEL.IF_WITH_CENTER != 'no':
        raise ValueError('This experiment retains the unsmoothed ID loss without center loss')
    for name in ('LPAM_PART_ID_MIX', 'LPAM_PART_TRIPLET_MIX'):
        if not 0 <= getattr(cfg.SOLVER, name) <= 1:
            raise ValueError(name+' must be in [0,1]')

    def loss_fn(parts, target):
        with torch.autocast(device_type=target.device.type, enabled=False):
            global_id = F.cross_entropy(parts['global_logits'].float(), target)
            positive_distance, negative_distance = hard_example_mining(
                euclidean_dist(parts['raw_readout'].float(), parts['raw_readout'].float()), target)
            global_tri = (F.softplus(positive_distance-negative_distance).mean() if cfg.MODEL.NO_MARGIN else
                          F.relu(positive_distance-negative_distance+cfg.SOLVER.MARGIN).mean())
            whole_id = F.cross_entropy(parts['local_logits'].float(), target)
            logits = parts['local_part_logits'].float()
            part_id = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                      target[:, None].expand(-1, logits.shape[1]).reshape(-1))
            id_mix = cfg.SOLVER.LPAM_PART_ID_MIX
            local_id = (1-id_mix)*whole_id+id_mix*part_id
            positive, negative = mining_masks(target, parts['mldc_after_bn'],
                                              cfg.SOLVER.LPAM_NEG_CANDIDATES,
                                              parts.get('camera_labels'))
            whole_tri = masked_triplet(parts['raw_lpam'], positive, negative, cfg.SOLVER.LPAM_MARGIN)
            normalized_parts = F.normalize(parts['raw_parts'].float(), dim=-1)
            part_tri = torch.stack([masked_triplet(normalized_parts[:, i], positive, negative,
                                                   cfg.SOLVER.LPAM_MARGIN)
                                    for i in range(normalized_parts.shape[1])]).mean()
            tri_mix = cfg.SOLVER.LPAM_PART_TRIPLET_MIX
            local_tri = (1-tri_mix)*whole_tri+tri_mix*part_tri
            # Forward values equal the main BN-after retrieval descriptor. The
            # detached global segment makes this added loss update the local
            # path, while the global path keeps its original ID/Triplet targets.
            fusion_tri = masked_triplet(parts['fusion_local_metric'], positive, negative,
                                        cfg.SOLVER.LPAM_FUSION_MARGIN)
            overlap = parts['overlap_loss'].float()
            total = cfg.MODEL.ID_LOSS_WEIGHT*global_id+cfg.MODEL.TRIPLET_LOSS_WEIGHT*global_tri
            total = total+cfg.SOLVER.LPAM_LOCAL_WEIGHT*(local_id+local_tri)
            total = total+cfg.SOLVER.LPAM_FUSION_WEIGHT*fusion_tri+cfg.SOLVER.LPAM_DIVERSITY_WEIGHT*overlap
            if not torch.isfinite(total).item():
                raise FloatingPointError('Non-finite LPAM v3 objective')
            loss_fn.last_terms = {name: value.detach() for name, value in
                [('global_id',global_id), ('global_triplet',global_tri),
                 ('local_id',local_id), ('local_triplet',local_tri),
                 ('local_whole_id',whole_id), ('local_part_id',part_id),
                 ('local_whole_triplet',whole_tri), ('local_part_triplet',part_tri),
                 ('fusion_triplet',fusion_tri), ('overlap',overlap)]}
            return total

    loss_fn.last_terms = {}
    return loss_fn
