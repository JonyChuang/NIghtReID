"""MLDC and free-query spatial LPAM, trained jointly from the first epoch."""
import torch
from torch import nn

from .backbones.vit_pytorch_mldc import vit_base_patch16_224_TransReID
from .mldc import MLDC
from .lpam import FreeQueryLPAM, unit_local, joint_descriptor

VARIANT = 'mldc_lpam_free_queries_v2'


def add_mldc_config(cfg):
    cfg.MODEL.MLDC_READOUT = 'mldc'
    cfg.MODEL.MLDC_TEMPERATURE = 0.9
    cfg.MODEL.LPAM_LAYERS = [6, 9, 12]
    cfg.MODEL.LPAM_DIM = 256
    cfg.MODEL.LPAM_QUERIES = 4
    cfg.MODEL.LPAM_HEADS = 4
    cfg.MODEL.LPAM_TEMPERATURE = 0.1
    cfg.SOLVER.LPAM_LOCAL_WEIGHT = 0.5
    cfg.SOLVER.LPAM_DIVERSITY_WEIGHT = 0.01
    cfg.SOLVER.LPAM_MARGIN = 0.3
    cfg.SOLVER.LPAM_NEG_CANDIDATES = 8
    cfg.SOLVER.LPAM_MIN_LR_RATIO = 0.01
    cfg.TEST.LPAM_WEIGHT = 0.2
    cfg.TEST.MLDC_VISUALIZE = True
    cfg.TEST.MLDC_VIS_SAMPLES = 6
    cfg.TEST.MLDC_VIS_PERIOD = 5
    cfg.TEST.LPAM_VIS_CASES_PER_GROUP = 2
    return cfg


class MLDCLPAMReID(nn.Module):
    def __init__(self, cfg, num_classes, camera_num, view_num, initialize=True):
        super().__init__()
        if cfg.MODEL.JPM or cfg.MODEL.GEM_POOLING or cfg.MODEL.USE_DCE or cfg.MODEL.QNORM_PRIOR or cfg.MODEL.QNORM_GUIDE:
            raise ValueError('Free-query LPAM requires the plain MLDC ViT backbone')
        if cfg.MODEL.TRANSFORMER_TYPE != 'vit_base_patch16_224_TransReID' or cfg.MODEL.MLDC_READOUT != 'mldc':
            raise ValueError('This experiment uses the Night600 MLDC ViT-Base')
        if cfg.TEST.NECK_FEAT != 'after':
            raise ValueError('Main retrieval uses independent BN necks; raw features are diagnostics')
        if cfg.MODEL.DIST_TRAIN or cfg.MODEL.REDUCE_FEAT_DIM or cfg.MODEL.ID_LOSS_TYPE != 'softmax':
            raise ValueError('This experiment uses single-GPU softmax with full global features')
        if cfg.MODEL.LPAM_QUERIES != 4:
            raise ValueError('This experiment and its visualization use four free part queries')
        self.readout, self.neck_feat = 'mldc', 'after'
        self.patch_layers = tuple(cfg.MODEL.LPAM_LAYERS)
        self.local_weight = float(cfg.TEST.LPAM_WEIGHT)
        self.checkpoint_epoch, self.phase = 0, 'joint'
        self.mldc = MLDC(12, cfg.MODEL.MLDC_TEMPERATURE)
        self.base = vit_base_patch16_224_TransReID(
            img_size=cfg.INPUT.SIZE_TRAIN, stride_size=cfg.MODEL.STRIDE_SIZE,
            camera=camera_num if cfg.MODEL.SIE_CAMERA else 0,
            view=view_num if cfg.MODEL.SIE_VIEW else 0,
            sie_xishu=cfg.MODEL.SIE_COE, stem_conv=cfg.MODEL.STEM_CONV,
            drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if initialize:
            if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
                # Keep the original generic SSL initialization, never an MLDC run.
                self.base.load_param(cfg.MODEL.PRETRAIN_PATH, hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
            elif cfg.MODEL.PRETRAIN_CHOICE != 'none':
                raise ValueError('Use imagenet for generic SSL initialization or none for random initialization')
        # These legacy modules are not used by the CLS/patch-token forward path.
        for module in (self.base.norm, self.base.fc, self.base.gem):
            module.requires_grad_(False)
        self.bottleneck = nn.BatchNorm1d(self.base.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.base.in_planes, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        self.dropout = nn.Dropout(cfg.MODEL.DROPOUT_RATE)
        self.lpam = FreeQueryLPAM(self.base.in_planes, cfg.MODEL.LPAM_DIM, self.patch_layers,
                                 cfg.MODEL.LPAM_QUERIES, cfg.MODEL.LPAM_HEADS, cfg.MODEL.LPAM_TEMPERATURE)
        self.local_necks = nn.ModuleList(nn.BatchNorm1d(cfg.MODEL.LPAM_DIM)
                                        for _ in range(cfg.MODEL.LPAM_QUERIES))
        for neck in self.local_necks:
            neck.bias.requires_grad_(False)
        self.local_classifier = nn.Linear(cfg.MODEL.LPAM_QUERIES*cfg.MODEL.LPAM_DIM, num_classes, bias=False)
        nn.init.normal_(self.local_classifier.weight, std=0.001)
        self.local_dropout = nn.Dropout(cfg.MODEL.DROPOUT_RATE)
        self.descriptor_dim = self.base.in_planes + cfg.MODEL.LPAM_QUERIES*cfg.MODEL.LPAM_DIM

    def optimizer_groups(self, cfg):
        groups = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            group_name = ('backbone' if name.startswith('base.') else
                          'global' if name.startswith(('mldc.', 'bottleneck.', 'classifier.')) else 'local')
            multiplier = cfg.SOLVER.BIAS_LR_FACTOR if 'bias' in name else 1.0
            decay = cfg.SOLVER.WEIGHT_DECAY_BIAS if 'bias' in name else cfg.SOLVER.WEIGHT_DECAY
            if cfg.SOLVER.LARGE_FC_LR and 'classifier' in name:
                multiplier = 2.0
            groups.append({'params': [parameter], 'name': group_name,
                           'lr': cfg.SOLVER.BASE_LR*multiplier,
                           'lr_multiplier': multiplier, 'weight_decay': decay})
        return groups

    def load_param(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint.get('variant') != VARIANT:
            raise ValueError('This evaluator requires a free-query LPAM v2 checkpoint')
        if checkpoint.get('architecture') != self.architecture():
            raise ValueError('Checkpoint architecture/attention temperature differs from configuration')
        self.load_state_dict(checkpoint['state_dict'], strict=True)
        self.checkpoint_epoch = int(checkpoint['epoch'])
        return checkpoint

    def architecture(self):
        return {'patch_layers': list(self.patch_layers), 'queries': self.lpam.queries,
                'dim': self.lpam.dim, 'heads': self.lpam.heads,
                'temperature': self.lpam.temperature, 'mldc_temperature': self.mldc.temperature,
                'descriptor_dim': self.descriptor_dim}

    def readout_weights(self):
        return self.mldc.normalized_weights()

    def forward(self, x, label=None, cam_label=None, view_label=None, return_components=False):
        backbone = self.base(x, cam_label=cam_label, view_label=view_label, patch_layers=self.patch_layers)
        layers = backbone['cls_layers']
        raw_global = self.mldc(layers)
        global_bn = self.bottleneck(raw_global)
        local = self.lpam(backbone['patch_layers'])
        raw_parts = local['parts']
        neck_parts = torch.stack([neck(raw_parts[:, i]) for i, neck in enumerate(self.local_necks)], dim=1)
        local_raw, local_bn = unit_local(raw_parts), unit_local(neck_parts)
        parts = {'retrieval': joint_descriptor(global_bn, local_bn, self.local_weight),
                 'mldc_after_bn': global_bn, 'lpam_after_bn': local_bn,
                 'raw_readout': raw_global, 'raw_last_cls': layers[-1], 'raw_uniform_cls': layers.mean(0),
                 'raw_lpam': local_raw, 'raw_parts': raw_parts, 'bn_parts': neck_parts,
                 'layers': layers, 'grid_hw': backbone['grid_hw'], **local}
        if self.training:
            parts['global_logits'] = self.classifier(self.dropout(global_bn))
            parts['local_logits'] = self.local_classifier(self.local_dropout(neck_parts.flatten(1)))
        if return_components or self.training:
            return parts
        return parts['retrieval']


def make_model(cfg, num_class, camera_num, view_num, initialize=True):
    return MLDCLPAMReID(cfg, num_class, camera_num, view_num, initialize)
