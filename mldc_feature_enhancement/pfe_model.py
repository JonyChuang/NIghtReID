"""Optional PFE model; existing B model and backbone files are left untouched."""
import torch
from torch import nn
from .model import FeatEnHancerMLDC, add_config as add_b_config, objectives

VARIANT = 'pfe_progressive_patch2468_mldc_fixed_equal_20261017'


def add_config(cfg):
    cfg = add_b_config(cfg)
    cfg.MODEL.PFE_ENABLED = True
    cfg.MODEL.PFE_LAYERS = [2, 4, 6, 8]
    cfg.MODEL.PFE_DIM = 192
    cfg.MODEL.PFE_KERNEL = 3
    cfg.MODEL.PFE_EXPANSION = .5
    cfg.MODEL.PFE_GAIN = .1
    cfg.TEST.PFE_VISUALIZE = True
    cfg.TEST.PFE_VIS_INDICES = [810, 1544, 11, 2059, 160, 24]
    return cfg


def config_from_dump(base_cfg, text):
    cfg = add_config(base_cfg.clone())
    stored = type(cfg).load_cfg(text)
    stored.MODEL.DEVICE_ID = repr(str(stored.MODEL.DEVICE_ID))
    cfg.merge_from_other_cfg(stored)
    return cfg


class PFEReID(FeatEnHancerMLDC):
    def __init__(self, cfg, num_classes, cameras, views, initialize=True):
        # Reuse the matched SSL ViT/head construction, with the B branch disabled.
        plain = cfg.clone(); plain.defrost()
        plain.MODEL.ENHANCEMENT_MODE = 'patch_feature'
        plain.MODEL.FEATENHANCER_ENABLED = False
        plain.freeze()
        super().__init__(plain, num_classes, cameras, views, initialize=initialize)
        if cfg.MODEL.ENHANCEMENT_MODE != 'pfe_progressive' or cfg.MODEL.FEATENHANCER_ENABLED:
            raise ValueError('PFE is a standalone enhancement experiment, without A/B branches')
        self.mode = 'pfe_progressive'
        self.enabled = bool(cfg.MODEL.PFE_ENABLED)
        self.pfe_layers = tuple(cfg.MODEL.PFE_LAYERS)
        if not self.pfe_layers or list(self.pfe_layers) != sorted(set(self.pfe_layers)) or not all(1 <= n < 9 for n in self.pfe_layers):
            raise ValueError('PFE depths must be sorted unique depths before global layers 9--12')
        self.pfe_settings = dict(dim=int(cfg.MODEL.PFE_DIM), kernel=int(cfg.MODEL.PFE_KERNEL),
                                 expansion=float(cfg.MODEL.PFE_EXPANSION), gain=float(cfg.MODEL.PFE_GAIN))
        self.grid = (self.base.patch_embed.num_y, self.base.patch_embed.num_x)
        self.pfe_adapters = nn.ModuleDict()
        self._pfe_diagnostics = False
        self._pfe_stats = {}
        self._cls_attention = {}
        self._pfe_handles = []
        if self.enabled:
            from .pfe_modules import PFETokenAdapter
            for depth in self.pfe_layers:
                self.pfe_adapters[str(depth)] = PFETokenAdapter(self.base.in_planes, **self.pfe_settings)
                self._pfe_handles.append(self.base.blocks[depth - 1].register_forward_hook(self._stage_hook(depth)))
        for depth in (9, 10, 11, 12):
            self._pfe_handles.append(self.base.blocks[depth - 1].attn.attn_drop.register_forward_hook(self._attention_hook(depth)))

    def _stage_hook(self, depth):
        def hook(module, args, output):
            if not self.enabled:
                return output
            enhanced, stats = self.pfe_adapters[str(depth)](output[:, 1:], self.grid, self._pfe_diagnostics)
            if self._pfe_diagnostics:
                self._pfe_stats[depth] = stats
            # Keep CLS unchanged here; the next Transformer reads changed patches.
            return torch.cat((output[:, :1], enhanced), 1)
        return hook

    def _attention_hook(self, depth):
        def hook(module, args, output):
            if self._pfe_diagnostics:
                self._cls_attention[depth] = output[:, :, 0, 1:].detach().float().mean(1).reshape(-1, *self.grid)
        return hook

    def forward(self, images, label=None, cam_label=None, view_label=None, diagnostics=False):
        self._pfe_stats = {}; self._cls_attention = {}
        self._pfe_diagnostics = bool(diagnostics)
        try:
            layers = self.base(images, cam_label=cam_label, view_label=view_label)
            raw = self.mldc(layers)
            neck = self.bottleneck(raw)
            if self.training or diagnostics:
                data = dict(retrieval=neck, raw_readout=raw, global_feature=raw, layers=layers, enhanced=images)
                if self.training:
                    data['scores'] = self.classifier(self.dropout(neck))
                if diagnostics:
                    data['cls_attention'] = torch.stack([self._cls_attention[l] for l in (9, 10, 11, 12)], 1)
                    if self.enabled:
                        for key in ('before_norm', 'after_norm', 'correction_norm', 'correction_ratio'):
                            data['pfe_' + key] = torch.stack([self._pfe_stats[l][key] for l in self.pfe_layers], 1)
                return data
            return neck
        finally:
            self._pfe_diagnostics = False
            self._pfe_stats = {}; self._cls_attention = {}

    def architecture(self):
        return dict(variant=VARIANT, enhancer='paper-based MSRFConv/PFEBlock in ViT, not the full detector',
                    pfe_enabled=self.enabled, pfe_layers=list(self.pfe_layers), **self.pfe_settings,
                    grid=list(self.grid), global_layers=[9, 10, 11, 12], global_fusion='fixed_equal',
                    global_weights=[.25] * 4, descriptor_dim=768, spatial_partition=False,
                    joint_from_epoch=1, retrieval='BN-after global; evaluator L2')
