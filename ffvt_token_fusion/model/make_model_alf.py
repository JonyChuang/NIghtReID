"""FFVT-style global/cross-layer ReID in the existing ffvt_token_fusion package."""
import torch
import torch.nn as nn

from .alf_module import CrossLayerFusion
from .backbones.vit_pytorch_alf import (vit_base_patch16_224_TransReID,
                                       vit_small_patch16_224_TransReID)

FACTORY = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
}


def add_alf_config(cfg):
    # Keep the existing entry-point import; only FFVT settings are now active.
    cfg.MODEL.FFVT_LAYERS = [3, 6, 9, 11]
    cfg.MODEL.FFVT_TOKENS_PER_LAYER = 12
    cfg.SOLVER.FFVT_HEAD_LR_FACTOR = 5.0
    cfg.TEST.NECK_FEAT = 'after'
    cfg.TEST.FEATURE_BRANCH = 'concat'
    return cfg


def configure_head_learning_rates(cfg, model, optimizer):
    """Only newly initialized BN/classification heads get the extra LR factor.

    Fusion block and fusion LayerNorm inherit the backbone optimizer settings,
    including its bias LR multiplier and weight decay.
    """
    prefixes = ('bottleneck_cls.', 'bottleneck_parts.',
                'classifier_cls.', 'classifier_parts.')
    head_parameters = {id(p) for name, p in model.named_parameters()
                       if p.requires_grad and name.startswith(prefixes)}
    matched = set()
    for group in optimizer.param_groups:
        ids = {id(p) for p in group['params']}
        if ids & head_parameters:
            if not ids <= head_parameters:
                raise RuntimeError('Optimizer mixes pretrained and new head parameters')
            group['lr'] *= cfg.SOLVER.FFVT_HEAD_LR_FACTOR
            matched.update(ids)
    if matched != head_parameters:
        raise RuntimeError('Some FFVT classification head parameters are missing from optimizer')


class FFVTReID(nn.Module):
    def __init__(self, cfg, num_classes, camera_num, view_num):
        super().__init__()
        if cfg.MODEL.NAME != 'transformer' or cfg.MODEL.TRANSFORMER_TYPE not in FACTORY:
            raise ValueError('FFVT experiment supports TransReID ViT/DeiT only')
        if cfg.MODEL.JPM or cfg.MODEL.GEM_POOLING or cfg.MODEL.USE_DCE:
            raise ValueError('FFVT requires the plain ViT backbone without JPM, GeM or DCE')
        if cfg.MODEL.ID_LOSS_TYPE != 'softmax' or cfg.MODEL.REDUCE_FEAT_DIM:
            raise ValueError('FFVT uses softmax ID heads and full feature dimensions')
        if cfg.MODEL.PRETRAIN_CHOICE not in ('imagenet', 'self'):
            raise ValueError('FFVT requires SSL backbone initialization or a compatible FFVT checkpoint')
        if cfg.TEST.NECK_FEAT not in ('before', 'after'):
            raise ValueError('TEST.NECK_FEAT must be before or after')
        if cfg.TEST.FEATURE_BRANCH not in ('concat', 'cls', 'fusion'):
            raise ValueError('TEST.FEATURE_BRANCH must be concat, cls or fusion')
        camera_num = camera_num if cfg.MODEL.SIE_CAMERA else 0
        view_num = view_num if cfg.MODEL.SIE_VIEW else 0
        self.base = FACTORY[cfg.MODEL.TRANSFORMER_TYPE](
            img_size=cfg.INPUT.SIZE_TRAIN, stride_size=cfg.MODEL.STRIDE_SIZE,
            sie_xishu=cfg.MODEL.SIE_COE, camera=camera_num, view=view_num,
            drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, stem_conv=cfg.MODEL.STEM_CONV,
            fusion_layers=cfg.MODEL.FFVT_LAYERS,
            tokens_per_layer=cfg.MODEL.FFVT_TOKENS_PER_LAYER)
        # Load SSL weights BEFORE copying the final block. The two branches
        # start with identical pretrained values and never share parameters.
        if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            self.base.load_param(cfg.MODEL.PRETRAIN_PATH,
                                 hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
        self.fusion = CrossLayerFusion(self.base.blocks[-1], self.base.norm)
        dim = self.base.in_planes
        # A single entry retains the existing independent-branch loss API.
        self.num_parts = 1
        self.part_dim = dim
        self.last_fusion_cls_ratio = None
        self.bottleneck_cls = self._make_neck(dim)
        self.bottleneck_parts = nn.ModuleList([self._make_neck(dim)])
        self.classifier_cls = self._make_classifier(dim, num_classes)
        self.classifier_parts = nn.ModuleList([self._make_classifier(dim, num_classes)])
        self.dropout = nn.Dropout(cfg.MODEL.DROPOUT_RATE)
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.feature_branch = cfg.TEST.FEATURE_BRANCH
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            self.load_param(cfg.MODEL.PRETRAIN_PATH)

    @staticmethod
    def _make_neck(dim):
        neck = nn.BatchNorm1d(dim)
        neck.bias.requires_grad_(False)
        nn.init.ones_(neck.weight)
        nn.init.zeros_(neck.bias)
        return neck

    @staticmethod
    def _make_classifier(dim, num_classes):
        classifier = nn.Linear(dim, num_classes, bias=False)
        nn.init.normal_(classifier.weight, std=0.001)
        return classifier

    def forward(self, x, label=None, cam_label=None, view_label=None,
                return_components=False, return_parts=False):
        patches, fusion_cls, final_cls = self.base(
            x, cam_label=cam_label, view_label=view_label)
        local_feature = self.fusion(fusion_cls, patches)
        parts = local_feature.unsqueeze(1)
        with torch.no_grad():
            self.last_fusion_cls_ratio = (
                local_feature.detach().float().norm(dim=1)
                / final_cls.detach().float().norm(dim=1).clamp_min(1e-12)
            ).mean()
        cls_bn = self.bottleneck_cls(final_cls)
        local_bn = self.bottleneck_parts[0](local_feature)
        if self.training:
            scores = {
                'cls': self.classifier_cls(self.dropout(cls_bn)),
                'parts': [self.classifier_parts[0](self.dropout(local_bn))],
            }
            return scores, {'cls': final_cls, 'parts': parts}
        global_output, local_output = ((cls_bn, local_bn) if self.neck_feat == 'after'
                                      else (final_cls, local_feature))
        if self.feature_branch == 'concat':
            output = torch.cat((global_output, local_output), dim=1)
        else:
            output = global_output if self.feature_branch == 'cls' else local_output
        if return_parts:
            return output, final_cls, local_feature, parts
        if return_components:
            return output, final_cls, local_feature
        return output

    def load_param(self, trained_path):
        state = torch.load(trained_path, map_location='cpu')
        state = state.get('state_dict', state.get('model', state))
        state = {key[7:] if key.startswith('module.') else key: value
                 for key, value in state.items()}
        if not any(key.startswith('fusion.block.') for key in state):
            raise ValueError('This is not a compatible FFVT checkpoint. Start the new experiment '
                             'from SSL with PRETRAIN_CHOICE imagenet; old ALF checkpoints cannot resume it.')
        self.load_state_dict(state, strict=True)
        print('Loaded FFVT checkpoint from {}'.format(trained_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = FFVTReID(cfg, num_classes=num_class, camera_num=camera_num, view_num=view_num)
    selected = len(cfg.MODEL.FFVT_LAYERS) * cfg.MODEL.FFVT_TOKENS_PER_LAYER
    print('FFVT dual branch: layers={}, selected patches={}, fusion input={} tokens, heads={}'.format(
        cfg.MODEL.FFVT_LAYERS, selected, selected + 1, model.fusion.block.attn.num_heads))
    print('Independent global/fusion ID + triplet; retrieval={}, neck={}, concat dimension={}'.format(
        model.feature_branch, model.neck_feat, 2 * model.base.in_planes))
    return model
