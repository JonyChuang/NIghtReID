"""MLDC baseline supervision, combined readout supervision, fusion metric."""
import torch
import torch.nn.functional as F
from loss.triplet_loss import euclidean_dist, hard_example_mining


def make_loss(cfg, num_classes=None):
    if cfg.MODEL.IF_LABELSMOOTH != 'off' or cfg.MODEL.IF_WITH_CENTER != 'no':
        raise ValueError('Retain the unsmoothed ID loss without center loss')

    def triplet(feature, target):
        positive, negative = hard_example_mining(
            euclidean_dist(feature.float(), feature.float()), target)
        if cfg.MODEL.NO_MARGIN:
            return F.softplus(positive-negative).mean()
        return F.relu(positive-negative+cfg.SOLVER.MARGIN).mean()

    def loss_fn(parts, target):
        with torch.autocast(device_type=target.device.type, enabled=False):
            global_id = F.cross_entropy(parts['global_logits'].float(), target)
            global_tri = triplet(parts['raw_readout'], target)
            read_id = F.cross_entropy(parts['read_logits'].float(), target)
            # Raw 3x256 reads before BN; neither per-layer ID loss nor
            # per-layer Triplet is imposed. Use the original PK batch mining.
            read_tri = triplet(parts['raw_local'], target)
            # This is the actual deployed BN-after weighted descriptor, with
            # gradients through both the global/read branches and fusion gate.
            fusion_tri = triplet(parts['retrieval'], target)
            total = cfg.MODEL.ID_LOSS_WEIGHT*global_id+cfg.MODEL.TRIPLET_LOSS_WEIGHT*global_tri
            total = total+cfg.SOLVER.LPAM_LOCAL_WEIGHT*(read_id+read_tri)
            total = total+cfg.SOLVER.LPAM_FUSION_WEIGHT*fusion_tri
            if not torch.isfinite(total).item():
                raise FloatingPointError('Non-finite MLDC-guided readout objective')
            loss_fn.last_terms = {name: value.detach() for name, value in
                [('global_id',global_id), ('global_triplet',global_tri),
                 ('read_id',read_id), ('read_triplet',read_tri), ('fusion_triplet',fusion_tri)]}
            return total

    loss_fn.last_terms = {}
    return loss_fn
