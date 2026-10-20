"""MLDC + TAR-refined last CLS, residual retrieval with three supervised heads."""
import torch
from torch import nn
from .backbones.vit_pytorch_mldc import vit_base_patch16_224_TransReID
from .mldc import MLDC
from .tar import TokenAwareRefinement

VARIANT = 'mldc_tar_residual_three_heads_v1'


def add_mldc_config(cfg):
    cfg.MODEL.MLDC_READOUT = 'mldc'
    cfg.MODEL.MLDC_TEMPERATURE = 0.9
    cfg.MODEL.TAR_TOP_RATIO = 0.5
    cfg.MODEL.TAR_NORM_ALPHA = 2.0
    cfg.MODEL.TAR_TEMPERATURE = 0.8
    cfg.MODEL.TAR_BOTTLENECK = 256
    cfg.MODEL.TAR_KL_BETA = 0.05
    cfg.SOLVER.MLDC_HEAD_LR_FACTOR = 5.0
    cfg.SOLVER.TAR_HEAD_LR_FACTOR = 5.0
    cfg.SOLVER.RESIDUAL_HEAD_LR_FACTOR = 5.0
    cfg.SOLVER.TAR_REID_WEIGHT = 0.5
    cfg.SOLVER.RESIDUAL_REID_WEIGHT = 1.0
    cfg.SOLVER.TAR_KL_WEIGHT = 0.001
    cfg.TEST.MLDC_VISUALIZE = True
    cfg.TEST.MLDC_VIS_SAMPLES = 6
    cfg.TEST.MLDC_VIS_PERIOD = 10
    return cfg


class MLDCTARReID(nn.Module):
    def __init__(self, cfg, num_classes, camera_num, view_num, initialize=True):
        super().__init__()
        if cfg.MODEL.JPM or cfg.MODEL.GEM_POOLING or cfg.MODEL.USE_DCE or cfg.MODEL.QNORM_PRIOR or cfg.MODEL.QNORM_GUIDE:
            raise ValueError('Use the plain MLDC backbone')
        if cfg.MODEL.TRANSFORMER_TYPE != 'vit_base_patch16_224_TransReID' or cfg.MODEL.MLDC_READOUT != 'mldc':
            raise ValueError('This experiment uses the Night600 MLDC ViT-Base')
        if cfg.TEST.NECK_FEAT != 'after' or cfg.MODEL.DIST_TRAIN or cfg.MODEL.REDUCE_FEAT_DIM or cfg.MODEL.ID_LOSS_TYPE != 'softmax':
            raise ValueError('Use single-GPU, full-dimension softmax and BN-after retrieval')
        self.readout, self.neck_feat, self.phase = 'mldc', 'after', 'joint'
        self.checkpoint_epoch, self.patch_layers, self.descriptor_dim = 0, (12,), 768
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
                self.base.load_param(cfg.MODEL.PRETRAIN_PATH, hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
            elif cfg.MODEL.PRETRAIN_CHOICE != 'none':
                raise ValueError('Use generic SSL initialization; resume through --resume')
        for module in (self.base.norm, self.base.fc, self.base.gem):
            module.requires_grad_(False)
        self.bottleneck, self.classifier = self._head(768, num_classes)
        self.tar = TokenAwareRefinement(768, cfg.MODEL.TAR_BOTTLENECK,
                                       cfg.MODEL.TAR_TOP_RATIO, cfg.MODEL.TAR_NORM_ALPHA,
                                       cfg.MODEL.TAR_TEMPERATURE, cfg.MODEL.TAR_KL_BETA)
        self.tar_neck, self.tar_classifier = self._head(768, num_classes)
        self.residual_neck, self.residual_classifier = self._head(768, num_classes)
        self.dropout = nn.Dropout(cfg.MODEL.DROPOUT_RATE)

    @staticmethod
    def _head(dim, classes):
        neck = nn.BatchNorm1d(dim)
        neck.bias.requires_grad_(False)
        classifier = nn.Linear(dim, classes, bias=False)
        nn.init.normal_(classifier.weight, std=0.001)
        return neck, classifier

    def optimizer_groups(self, cfg):
        groups = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith(('bottleneck.', 'classifier.')):
                group, factor = 'global_head', cfg.SOLVER.MLDC_HEAD_LR_FACTOR
            elif name.startswith(('tar_neck.', 'tar_classifier.')):
                group, factor = 'tar_head', cfg.SOLVER.TAR_HEAD_LR_FACTOR
            elif name.startswith(('residual_neck.', 'residual_classifier.')):
                group, factor = 'residual_head', cfg.SOLVER.RESIDUAL_HEAD_LR_FACTOR
            else:
                group, factor = ('backbone' if name.startswith('base.') else name.split('.')[0]), 1.0
            bias = 'bias' in name
            multiplier = cfg.SOLVER.BIAS_LR_FACTOR if bias else 1.0
            if cfg.SOLVER.LARGE_FC_LR and 'classifier' in name:
                multiplier = 2.0
            groups.append({'params': [parameter], 'name': group + ('_bias' if bias else ''),
                           'lr': cfg.SOLVER.BASE_LR * multiplier * factor,
                           'weight_decay': cfg.SOLVER.WEIGHT_DECAY_BIAS if bias else cfg.SOLVER.WEIGHT_DECAY})
        return groups

    def architecture(self):
        return {'variant': VARIANT, 'patch_layers': [12], 'descriptor_dim': 768,
                'mldc_temperature': self.mldc.temperature,
                'tar_top_ratio': self.tar.top_ratio, 'tar_norm_alpha': self.tar.norm_alpha,
                'tar_temperature': self.tar.temperature, 'tar_bottleneck': self.tar.bottleneck,
                'tar_kl_beta': self.tar.kl_beta, 'fusion': 'raw_mldc_plus_refined_last_cls',
                'heads': ['mldc', 'tar_cls', 'residual']}

    def load_param(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint.get('variant') != VARIANT or checkpoint.get('architecture') != self.architecture():
            raise ValueError('Require a matching MLDC + TAR residual three-head checkpoint')
        self.load_state_dict(checkpoint['state_dict'], strict=True)
        self.checkpoint_epoch = int(checkpoint['epoch'])
        return checkpoint

    def readout_weights(self):
        return self.mldc.normalized_weights()

    def forward(self, x, label=None, cam_label=None, view_label=None, return_components=False):
        backbone = self.base(x, cam_label=cam_label, view_label=view_label, patch_layers=(12,))
        layers = backbone['cls_layers']
        raw_global = self.mldc(layers)
        tar = self.tar(layers[-1], backbone['patch_layers'][:, 0])
        refined = tar['refined_cls']
        # Add the full refined CLS, including its original CLS residual.
        raw_residual = raw_global.float() + refined
        global_bn = self.bottleneck(raw_global)
        tar_bn = self.tar_neck(refined)
        residual_bn = self.residual_neck(raw_residual)
        parts = {'raw_readout': raw_global, 'mldc_after_bn': global_bn,
                 'raw_last_cls': layers[-1], 'raw_tar': refined, 'tar_after_bn': tar_bn,
                 'raw_residual': raw_residual, 'retrieval': residual_bn,
                 'layers': layers, 'grid_hw': backbone['grid_hw'], **tar}
        if self.training:
            parts['global_logits'] = self.classifier(self.dropout(global_bn))
            parts['tar_logits'] = self.tar_classifier(self.dropout(tar_bn))
            parts['residual_logits'] = self.residual_classifier(self.dropout(residual_bn))
        return parts if return_components or self.training else residual_bn


def make_model(cfg, num_class, camera_num, view_num, initialize=True):
    return MLDCTARReID(cfg, num_class, camera_num, view_num, initialize)
