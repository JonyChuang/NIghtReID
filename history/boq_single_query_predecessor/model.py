"""Joint training with a single local descriptor, no spatial partitions."""
import torch
from torch import nn
import torch.nn.functional as F


class MiddlePatchReader(nn.Module):
    def __init__(self, input_dim=768, dim=192, heads=4, layers=(4, 6, 8), mode='cosine', temperature=0.2):
        super().__init__()
        if dim % heads or temperature <= 0 or mode not in ('cosine', 'mean'):
            raise ValueError('Invalid local reader configuration')
        self.layers = tuple(layers)
        if not self.layers or tuple(sorted(set(self.layers))) != self.layers or any(i < 1 or i > 12 for i in self.layers):
            raise ValueError('Layers must be increasing and unique in [1,12]')
        self.dim, self.heads, self.mode, self.temperature = dim, heads, mode, temperature
        self.norms = nn.ModuleList(nn.LayerNorm(input_dim) for _ in self.layers)
        self.values = nn.ModuleList(nn.Linear(input_dim, dim, bias=False) for _ in self.layers)
        if mode == 'cosine':
            self.keys = nn.ModuleList(nn.Linear(input_dim, dim, bias=False) for _ in self.layers)
            self.query = nn.Parameter(torch.empty(len(self.layers), heads, dim // heads))
            nn.init.normal_(self.query, std=0.02)
        self.output = nn.Sequential(nn.Linear(dim, input_dim, bias=False), nn.LayerNorm(input_dim))

    def forward(self, patches):
        if len(patches) != len(self.layers):
            raise ValueError('One patch tensor is required for each selected layer')
        summaries, maps = [], []
        for i, (norm, value, patch) in enumerate(zip(self.norms, self.values, patches)):
            x = norm(patch)
            b, n, _ = x.shape
            v = value(x).reshape(b, n, self.heads, -1).transpose(1, 2)
            k = self.keys[i](x).reshape(b, n, self.heads, -1).transpose(1, 2) if self.mode == 'cosine' else None
            with torch.autocast(device_type=x.device.type, enabled=False):
                if self.mode == 'cosine':
                    q = F.normalize(self.query[i].float(), dim=-1, eps=1e-6)
                    k = F.normalize(k.float(), dim=-1, eps=1e-6)
                    logits = torch.einsum('hd,bhnd->bhn', q, k) / self.temperature
                    attention = logits.softmax(-1)
                else:
                    attention = v.new_full((b, self.heads, n), 1.0 / n, dtype=torch.float32)
                summary = torch.einsum('bhn,bhnd->bhd', attention, v.float())
            summaries.append(summary.reshape(b, self.dim))
            maps.append(attention)
        # Equal depth weights keep depth choice separate from spatial selection.
        pooled = torch.stack(summaries, 1).mean(1)
        local = self.output(pooled.to(patches[0].dtype))
        return local, torch.stack(maps, 2)  # B, heads, depths, patches


def fuse_local(global_feature, local_feature, strength=0.1):
    # Every scale and direction remains differentiable during joint training.
    with torch.autocast(device_type=global_feature.device.type, enabled=False):
        g, a = global_feature.float(), local_feature.float()
        delta = strength * g.norm(dim=-1, keepdim=True) * F.normalize(a, dim=-1, eps=1e-6)
    return global_feature + delta.to(global_feature.dtype), delta


class JointLocalReID(nn.Module):
    def __init__(self, global_model, num_classes, layers=(4, 6, 8), dim=192, heads=4,
                 mode='cosine', temperature=0.2, strength=0.1):
        super().__init__()
        if not 0 < strength <= 1:
            raise ValueError('Local fusion strength must lie in (0,1]')
        self.global_model = global_model
        self.reader = MiddlePatchReader(global_model.base.in_planes, dim, heads, layers, mode, temperature)
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
        return {'version': 1, 'joint_from_epoch': 1, 'cls_layers': [9,10,11,12],
                'patch_layers': list(self.reader.layers), 'reader': self.reader.mode,
                'dim': self.reader.dim, 'heads': self.reader.heads, 'temperature': self.reader.temperature,
                'fusion_strength': self.strength, 'spatial_partition': False,
                'depth_weights': 'fixed_equal', 'descriptor_dim': 768}


def add_config(cfg):
    from mldc_deep4_readout.model.make_model_mldc import add_mldc_config
    cfg = add_mldc_config(cfg)
    cfg.MODEL.LOCAL_LAYERS = [4, 6, 8]
    cfg.MODEL.LOCAL_DIM = 192
    cfg.MODEL.LOCAL_HEADS = 4
    cfg.MODEL.LOCAL_READER = 'cosine'
    cfg.MODEL.LOCAL_TEMPERATURE = 0.2
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
                          cfg.MODEL.LOCAL_DIM, cfg.MODEL.LOCAL_HEADS, cfg.MODEL.LOCAL_READER,
                          cfg.MODEL.LOCAL_TEMPERATURE, cfg.MODEL.LOCAL_STRENGTH)
