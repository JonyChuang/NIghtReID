"""Independent patch ID/triplet supervision plus the unchanged fused objective."""
import math
import torch
import torch.nn.functional as F
from loss.center_loss import CenterLoss
from loss.softmax_loss import CrossEntropyLabelSmooth
from loss.triplet_loss import euclidean_dist, hard_example_mining


def make_loss(cfg, num_classes):
    if (cfg.DATALOADER.SAMPLER != 'softmax_triplet'
            or cfg.MODEL.METRIC_LOSS_TYPE != 'triplet' or cfg.MODEL.IF_WITH_CENTER != 'no'):
        raise ValueError('Use identity + triplet without center loss')
    weight=float(cfg.SOLVER.PATCH_LOCAL_LOSS_WEIGHT)
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError('Local supervision weight must be finite and positive')
    smoothed=CrossEntropyLabelSmooth(num_classes=num_classes) if cfg.MODEL.IF_LABELSMOOTH=='on' else None
    def identity(scores,target):
        return smoothed(scores,target) if smoothed is not None else F.cross_entropy(scores,target)
    def triplet(feature,target):
        positive,negative=hard_example_mining(euclidean_dist(feature,feature),target)
        gap=positive-negative
        return F.softplus(gap).mean() if cfg.MODEL.NO_MARGIN else F.relu(gap+cfg.SOLVER.MARGIN).mean()
    def loss_fn(outputs,target,target_cam=None):
        with torch.autocast(device_type=outputs['local_feature'].device.type,enabled=False):
            local_id=identity(outputs['local_scores'].float(),target)
            local_tri=triplet(outputs['local_feature'].float(),target)
            local=cfg.MODEL.ID_LOSS_WEIGHT*local_id+cfg.MODEL.TRIPLET_LOSS_WEIGHT*local_tri
            if outputs['stage']=='local_warmup':
                main_id=local_id.detach().new_zeros(())
                main_tri=local_tri.detach().new_zeros(())
                total=local
            elif outputs['stage']=='joint_fusion':
                main_id=identity(outputs['scores'].float(),target)
                main_tri=triplet(outputs['feature'].float(),target)
                total=(cfg.MODEL.ID_LOSS_WEIGHT*main_id+
                       cfg.MODEL.TRIPLET_LOSS_WEIGHT*main_tri+weight*local)
            else:
                raise ValueError('Unknown patch training stage')
            if not torch.isfinite(total).item():
                raise FloatingPointError('Non-finite patch-GRM loss')
            loss_fn.last_terms={k:v.detach() for k,v in dict(identity=main_id,triplet=main_tri,
                                                            local_identity=local_id,local_triplet=local_tri).items()}
            return total
    loss_fn.last_terms={}
    return loss_fn,CenterLoss(num_classes=num_classes,feat_dim=768,use_gpu=False)
