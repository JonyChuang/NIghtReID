"""Joint deep-four CLS fusion and multi-query BoQ patch aggregation."""
import torch
from torch import nn
import torch.nn.functional as F
from .boq import BoQPatchReader

EXPERIMENT = 'mldc_boq_joint_v2_20261017'


def fuse_local(global_feature, local_feature, strength=0.1):
    # Every scale and direction remains differentiable during joint training.
    with torch.autocast(device_type=global_feature.device.type, enabled=False):
        g, a = global_feature.float(), local_feature.float()
        delta = strength * g.norm(dim=-1, keepdim=True) * F.normalize(a, dim=-1, eps=1e-6)
    return global_feature + delta.to(global_feature.dtype), delta


class JointLocalReID(nn.Module):
    def __init__(self, global_model, num_classes, layers=(4, 6, 8), dim=192, heads=4,
                 queries=4, row_dim=None, strength=0.1):
        super().__init__()
        if not 0 < strength <= 1:
            raise ValueError('Local fusion strength must lie in (0,1]')
        self.global_model = global_model
        self.reader = BoQPatchReader(global_model.base.in_planes, dim, heads, layers, queries, row_dim)
        self.strength = strength
        self.local_bn = nn.BatchNorm1d(global_model.base.in_planes)
        self.local_bn.bias.requires_grad_(False)  # Standard BN-neck constant offset.
        self.local_classifier = nn.Linear(global_model.base.in_planes, num_classes, bias=False)
        nn.init.normal_(self.local_classifier.weight, std=0.001)
        if not all(p.requires_grad for block in global_model.base.blocks for p in block.parameters()):
            raise ValueError('The active backbone blocks must participate in joint training')
        if not global_model.mldc.layer_weights.requires_grad:
            raise ValueError('MLDC fusion weights must participate in joint training')

    def forward(self, x, label=None, cam_label=None, view_label=None, diagnostics=False):
        captured = {}
        handles = []
        try:
            for layer in self.reader.layers:
                def capture(module, args, output, layer=layer):
                    captured[layer] = output[:, 1:]  # Post-block image tokens; full gradient.
                handles.append(self.global_model.base.blocks[layer - 1].register_forward_hook(capture))
            cls = self.global_model.base(x, cam_label=cam_label, view_label=view_label)
        finally:
            for handle in handles:
                handle.remove()
        g = self.global_model.mldc(cls)
        a, attention = self.reader([captured[layer] for layer in self.reader.layers])
        f, delta = fuse_local(g, a, self.strength)
        final_bn = self.global_model.bottleneck(f)
        local_bn = self.local_bn(a)
        result = {'feature': f, 'local_feature': a, 'global_feature': g,
                  'output': final_bn if self.global_model.neck_feat == 'after' else f,
                  'local_bn': local_bn, 'attention': attention, 'delta': delta}
        if self.training:
            result['scores'] = self.global_model.classifier(self.global_model.dropout(final_bn))
            result['local_scores'] = self.local_classifier(self.global_model.dropout(local_bn))
            return result
        if diagnostics:
            # This BN was trained on f; the following is only a removal counterfactual.
            result['global_same_final_bn'] = self.global_model.bottleneck(g)
            result['cls'] = cls
            return result
        return result['output']

    def architecture(self):
        return {'version': 2, 'joint_from_epoch': 1, 'cls_layers': [9,10,11,12],
                'patch_layers': list(self.reader.layers), 'reader': self.reader.mode,
                'dim': self.reader.dim, 'heads': self.reader.heads, 'queries_per_depth': self.reader.queries,
                'row_dim': self.reader.row_dim, 'patch_encoder': True, 'query_self_attention': True,
                'query_residual_to_output': False, 'independent_bag_per_depth': True,
                'fusion_strength': self.strength, 'spatial_partition': False,
                'aggregation': 'concat_depth_query_then_learned_slot_projection',
                'descriptor_dim': self.global_model.base.in_planes}


def add_config(cfg):
    from mldc_deep4_readout.model.make_model_mldc import add_mldc_config
    cfg = add_mldc_config(cfg)
    cfg.MODEL.LOCAL_LAYERS = [4, 6, 8]
    cfg.MODEL.LOCAL_DIM = 192
    cfg.MODEL.LOCAL_HEADS = 4
    cfg.MODEL.LOCAL_READER = 'boq'
    cfg.MODEL.BOQ_QUERIES = 4
    cfg.MODEL.BOQ_ROW_DIM = 4
    cfg.MODEL.LOCAL_STRENGTH = 0.1
    cfg.SOLVER.LOCAL_LOSS_WEIGHT = 0.1
    return cfg


def build_model(cfg, num_classes, cameras, views, initialize=True):
    from mldc_deep4_readout.model.make_model_mldc import MLDCReID
    if (cfg.MODEL.PRETRAIN_CHOICE != 'imagenet' or cfg.MODEL.MLDC_INIT != 'uniform'
            or list(cfg.MODEL.MLDC_LAYERS) != [9,10,11,12]
            or cfg.MODEL.MLDC_TRAIN_FEATURE != 'mldc' or cfg.MODEL.MLDC_READOUT != 'mldc'
            or cfg.TEST.MLDC_RETRIEVAL != 'mldc' or not cfg.MODEL.NO_MARGIN
            or cfg.TEST.NECK_FEAT != 'after'
            or cfg.MODEL.LOCAL_READER != 'boq'
            or cfg.MODEL.IF_LABELSMOOTH != 'off'):
        raise ValueError('Use fresh SSL initialization and the matched deep-four baseline setup')
    if initialize:
        global_model = MLDCReID(cfg, num_classes, cameras, views)
    else:
        class WithoutInitialization(MLDCReID):
            def load_param(self, path):
                pass
        constructor_cfg = cfg.clone()
        constructor_cfg.defrost()
        constructor_cfg.MODEL.PRETRAIN_CHOICE = 'self'
        constructor_cfg.freeze()
        global_model = WithoutInitialization(constructor_cfg, num_classes, cameras, views)
    return JointLocalReID(global_model, num_classes, cfg.MODEL.LOCAL_LAYERS,
                          cfg.MODEL.LOCAL_DIM, cfg.MODEL.LOCAL_HEADS, cfg.MODEL.BOQ_QUERIES,
                          cfg.MODEL.BOQ_ROW_DIM, cfg.MODEL.LOCAL_STRENGTH)
