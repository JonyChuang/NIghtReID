"""Independent GLTrans experiment on the original TransReID-SSL ViT."""
import copy

import torch
from torch import nn

from .backbones.vit_pytorch_gltrans_night import (vit_base_patch16_224_TransReID,
                                                  vit_small_patch16_224_TransReID)
from .gltrans_modules import GLTransModules


FACTORY = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
}


def add_gltrans_config(cfg):
    cfg.MODEL.GLTRANS_LAYERS = [10, 11, 12]
    cfg.MODEL.GLTRANS_PARTS = 3
    return cfg


def _init_classifier(module):
    nn.init.normal_(module.weight, std=0.001)


class GLTransNight(nn.Module):
    """GAE + PAM/PTF/GMA/PTL with separately supervised global/local outputs."""

    def __init__(self, num_classes, camera_num, view_num, cfg):
        super().__init__()
        if cfg.MODEL.NAME != 'transformer' or cfg.MODEL.TRANSFORMER_TYPE not in FACTORY:
            raise ValueError('GLTrans Night supports the TransReID ViT/DeiT backbone')
        if cfg.MODEL.JPM or cfg.MODEL.GEM_POOLING:
            raise ValueError('GLTrans Night uses its own local branch instead of JPM/GeM')
        self.layers = tuple(cfg.MODEL.GLTRANS_LAYERS)
        if self.layers != (10, 11, 12) or cfg.MODEL.GLTRANS_PARTS != 3:
            raise ValueError('The official GLTrans structure uses blocks 10/11/12 and three parts')
        camera_num = camera_num if cfg.MODEL.SIE_CAMERA else 0
        view_num = view_num if cfg.MODEL.SIE_VIEW else 0
        self.base = FACTORY[cfg.MODEL.TRANSFORMER_TYPE](
            img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
            camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
            stem_conv=cfg.MODEL.STEM_CONV, gem_pool=False)
        if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            self.base.load_param(cfg.MODEL.PRETRAIN_PATH,
                                 hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
        dim = self.base.in_planes
        heads = self.base.blocks[0].attn.num_heads
        self.token_norms = nn.ModuleList([copy.deepcopy(self.base.norm)
                                          for _ in self.layers])
        # Those original heads are retained in the copied backbone only for
        # checkpoint compatibility; GLTrans trains its own norms and heads.
        for unused in (self.base.norm, self.base.fc, self.base.gem):
            for parameter in unused.parameters():
                parameter.requires_grad_(False)
        self.gltrans = GLTransModules(dim, heads, cfg.MODEL.GLTRANS_PARTS)
        self.bn_cls = nn.BatchNorm1d(dim)
        self.bn_global = nn.BatchNorm1d(dim)
        self.bn_local = nn.BatchNorm1d(3 * dim)
        self.bn_aux = nn.ModuleList(nn.BatchNorm1d(dim) for _ in range(2))
        self.classifier_cls = nn.Linear(dim, num_classes, bias=False)
        self.classifier_global = nn.Linear(dim, num_classes, bias=False)
        self.classifier_local = nn.Linear(3 * dim, num_classes, bias=False)
        self.classifier_aux = nn.ModuleList(
            nn.Linear(dim, num_classes, bias=False) for _ in range(2))
        for bn in [self.bn_cls, self.bn_global, self.bn_local, *self.bn_aux]:
            nn.init.ones_(bn.weight)
            nn.init.zeros_(bn.bias)
        for classifier in [self.classifier_cls, self.classifier_global,
                           self.classifier_local, *self.classifier_aux]:
            _init_classifier(classifier)
        self.descriptor_dim = 5 * dim
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            self.load_param(cfg.MODEL.PRETRAIN_PATH)

    def _tokens(self, image, camera_id, view_id):
        base = self.base
        batch = image.shape[0]
        patches = base.patch_embed(image)
        grid_size = (base.patch_embed.num_y, base.patch_embed.num_x)
        if patches.shape[1] != grid_size[0] * grid_size[1]:
            raise ValueError('Actual patch grid differs from configured image size')
        cls = base.cls_token.expand(batch, -1, -1)
        tokens = torch.cat((cls, patches), dim=1)
        if base.cam_num > 0 and base.view_num > 0:
            tokens = tokens + base.pos_embed + base.sie_xishu * base.sie_embed[
                camera_id * base.view_num + view_id]
        elif base.cam_num > 0:
            tokens = tokens + base.pos_embed + base.sie_xishu * base.sie_embed[camera_id]
        elif base.view_num > 0:
            tokens = tokens + base.pos_embed + base.sie_xishu * base.sie_embed[view_id]
        else:
            tokens = tokens + base.pos_embed
        tokens = base.pos_drop(tokens)
        patch_tokens = []
        class_tokens = []
        for block_number, block in enumerate(base.blocks, start=1):
            tokens = block(tokens)
            if block_number in self.layers:
                patch_tokens.append(tokens[:, 1:])
                class_tokens.append(self.token_norms[len(class_tokens)](tokens)[:, 0])
        if len(class_tokens) != 3:
            raise RuntimeError('Did not collect all three GLTrans layers')
        return patch_tokens, torch.stack(class_tokens, dim=1), grid_size

    def forward(self, image, label=None, cam_label=None, view_label=None):
        patches, class_tokens, grid_size = self._tokens(image, cam_label, view_label)
        global_feature, local_feature = self.gltrans(patches, class_tokens, grid_size)
        final_cls = class_tokens[:, -1]
        if not self.training:
            # Official GLTrans descriptor ordering and dimensionality.
            return torch.cat((local_feature, global_feature, final_cls), dim=1)
        scores = [
            self.classifier_cls(self.bn_cls(final_cls)),
            self.classifier_global(self.bn_global(global_feature)),
            self.classifier_local(self.bn_local(local_feature)),
            self.classifier_aux[0](self.bn_aux[0](class_tokens[:, 0])),
            self.classifier_aux[1](self.bn_aux[1](class_tokens[:, 1])),
        ]
        features = [final_cls, global_feature, local_feature]
        return scores, features

    def load_param(self, trained_path):
        state = torch.load(trained_path, map_location='cpu')
        if 'state_dict' in state:
            state = state['state_dict']
        elif 'model' in state:
            state = state['model']
        state = {key[7:] if key.startswith('module.') else key: value
                 for key, value in state.items()}
        self.load_state_dict(state, strict=True)
        print('Loaded GLTrans Night checkpoint from {}'.format(trained_path))


def make_model(cfg, num_class, camera_num, view_num):
    return GLTransNight(num_class, camera_num, view_num, cfg)
