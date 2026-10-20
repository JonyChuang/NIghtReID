"""Fused ReID objective plus weak, direct local ReID supervision."""
import torch
import torch.nn.functional as F
from loss.triplet_loss import euclidean_dist, hard_example_mining


def objectives(outputs, targets, local_weight=0.1):
    def triplet(feature):
        positive, negative = hard_example_mining(euclidean_dist(feature, feature), targets)
        return F.softplus(positive - negative).mean()
    with torch.autocast(device_type=outputs['feature'].device.type, enabled=False):
        terms = {'identity': F.cross_entropy(outputs['scores'].float(), targets),
                 'triplet': triplet(outputs['feature'].float()),
                 'local_identity': F.cross_entropy(outputs['local_scores'].float(), targets),
                 'local_triplet': triplet(outputs['local_feature'].float())}
        total = terms['identity'] + terms['triplet'] + local_weight * (terms['local_identity'] + terms['local_triplet'])
    return total, terms
