"""Original TransReID-SSL global branch, with an optional CNL descriptor return."""
import torch
from torch import nn
from torch.nn import functional as F
from .backbone import vit_base_patch16_224_TransReID


def add_cnl_config(cfg):
    cfg.SOLVER.CNL_WEIGHT = 0.1
    cfg.SOLVER.CNL_GAMMA = 0.9
    cfg.SOLVER.CNL_TEMPERATURE = 0.05
    cfg.SOLVER.CNL_MOMENTUM = 0.2
    return cfg


class TransReIDSSLCNL(nn.Module):
    def __init__(self, cfg, num_classes, camera_num, view_num):
        super().__init__()
        if cfg.MODEL.NAME != 'transformer' or cfg.MODEL.TRANSFORMER_TYPE != 'vit_base_patch16_224_TransReID':
            raise ValueError('Use the original Night600 ViT-Base TransReID-SSL backbone')
        if any((cfg.MODEL.JPM, cfg.MODEL.GEM_POOLING, cfg.MODEL.USE_DCE,
                cfg.MODEL.QNORM_PRIOR, cfg.MODEL.QNORM_GUIDE, cfg.MODEL.REDUCE_FEAT_DIM,
                cfg.MODEL.DIST_TRAIN)) or cfg.MODEL.ID_LOSS_TYPE != 'softmax':
            raise ValueError('This package implements the original single-GPU global SSL branch')
        if cfg.MODEL.PRETRAIN_CHOICE not in ('imagenet', 'self'):
            raise ValueError('Use imagenet for SSL initialization or self for a saved experiment')
        if cfg.TEST.NECK_FEAT not in ('before', 'after'):
            raise ValueError('NECK_FEAT must be before or after')
        self.base = vit_base_patch16_224_TransReID(
            img_size=cfg.INPUT.SIZE_TRAIN, stride_size=cfg.MODEL.STRIDE_SIZE,
            camera=camera_num if cfg.MODEL.SIE_CAMERA else 0,
            view=view_num if cfg.MODEL.SIE_VIEW else 0,
            sie_xishu=cfg.MODEL.SIE_COE, stem_conv=cfg.MODEL.STEM_CONV,
            drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, gem_pool=False)
        if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            self.base.load_param(cfg.MODEL.PRETRAIN_PATH, hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
        self.in_planes = self.base.in_planes
        # Preserve the original construction order and initialization.
        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.ones_(self.bottleneck.weight)
        nn.init.zeros_(self.bottleneck.bias)
        self.dropout = nn.Dropout(cfg.MODEL.DROPOUT_RATE)
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.checkpoint_epoch = 0
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            self.load_param(cfg.MODEL.PRETRAIN_PATH)

    def forward(self, x, label=None, cam_label=None, view_label=None, return_cnl_features=False):
        # The backbone includes its original final LayerNorm before taking CLS.
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)
        neck = self.bottleneck(global_feat)
        retrieval = neck if self.neck_feat == 'after' else global_feat
        if self.training:
            scores = self.classifier(self.dropout(neck))
            if return_cnl_features:
                return scores, global_feat, F.normalize(retrieval.float(), dim=1)
            return scores, global_feat
        return retrieval

    def load_param(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        if 'variant' in checkpoint and checkpoint['variant'] != 'transreid_ssl_cnl_20261022':
            raise ValueError('Checkpoint is from a different experiment')
        state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        self.load_state_dict({k[7:] if k.startswith('module.') else k: v for k, v in state.items()}, strict=True)
        self.checkpoint_epoch = int(checkpoint.get('epoch', 0))


def make_model(cfg, num_class, camera_num, view_num):
    return TransReIDSSLCNL(cfg, num_class, camera_num, view_num)
