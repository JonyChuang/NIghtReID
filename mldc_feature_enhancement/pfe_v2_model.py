"""PFE v2: measurable image-wide residual magnitude, same PFE core/backbone."""
import torch
from torch import nn
from .pfe_modules import PFETokenAdapter
from .pfe_model import PFEReID, add_config as add_pfe_config

VARIANT = 'pfe_v2_rms5pct_lr10_patch2468_fixed_equal_20261018'


def add_config(cfg):
    cfg = add_pfe_config(cfg)
    cfg.MODEL.PFE_TARGET_RATIO = .05
    cfg.MODEL.PFE_UP_INIT_STD = .01
    cfg.SOLVER.PFE_LR_FACTOR = 10.
    return cfg


def config_from_dump(base_cfg, text):
    cfg = add_config(base_cfg.clone())
    stored = type(cfg).load_cfg(text)
    stored.MODEL.DEVICE_ID = repr(str(stored.MODEL.DEVICE_ID))
    cfg.merge_from_other_cfg(stored)
    return cfg


class RMSPFETokenAdapter(PFETokenAdapter):
    def __init__(self, input_dim=768, dim=192, kernel=3, expansion=.5,
                 target_ratio=.05, up_init_std=.01):
        super().__init__(input_dim, dim, kernel, expansion, gain=1.)
        if not 0 < target_ratio <= .05 or up_init_std <= 0:
            raise ValueError('Use a positive residual ratio <=5% and positive up initialization')
        self.target_ratio = float(target_ratio)
        nn.init.normal_(self.up.weight, std=up_init_std)

    def forward(self, patches, grid, diagnostics=False):
        b, n, _ = patches.shape
        if n != grid[0] * grid[1]:
            raise ValueError('Patch sequence does not match grid')
        spatial = self.down(self.norm(patches)).transpose(1, 2).reshape(b, -1, *grid)
        raw = self.up(self.block.increment(spatial).flatten(2).transpose(1, 2))
        # One scale per image over ALL tokens/channels preserves spatial contrast.
        # Keep both normalizers and addition in FP32, also under AMP. No detach:
        # backbone, normalization and the entire PFE direction train jointly.
        with torch.autocast(device_type=patches.device.type, enabled=False):
            original, direction = patches.float(), raw.float()
            source_rms = original.square().mean((1, 2), keepdim=True).sqrt()
            delta_rms = direction.square().mean((1, 2), keepdim=True).clamp_min(1e-16).sqrt()
            correction = self.target_ratio * source_rms * direction / delta_rms
            enhanced = original + correction
        stats = {}
        if diagnostics:
            before, after, delta = original.detach(), enhanced.detach(), correction.detach()
            stats = dict(before_norm=before.norm(dim=-1).reshape(b, *grid),
                         after_norm=after.norm(dim=-1).reshape(b, *grid),
                         correction_norm=delta.norm(dim=-1).reshape(b, *grid),
                         correction_ratio=delta.flatten(1).norm(dim=-1) / before.flatten(1).norm(dim=-1).clamp_min(1e-12))
        return enhanced, stats


class PFEV2ReID(PFEReID):
    def __init__(self, cfg, num_classes, cameras, views, initialize=True):
        # Construct the original SSL backbone and heads before creating adapters.
        plain = cfg.clone(); plain.defrost(); plain.MODEL.PFE_ENABLED = False; plain.freeze()
        super().__init__(plain, num_classes, cameras, views, initialize=initialize)
        self.enabled = bool(cfg.MODEL.PFE_ENABLED)
        self.pfe_settings = dict(dim=int(cfg.MODEL.PFE_DIM), kernel=int(cfg.MODEL.PFE_KERNEL),
            expansion=float(cfg.MODEL.PFE_EXPANSION), target_ratio=float(cfg.MODEL.PFE_TARGET_RATIO),
            up_init_std=float(cfg.MODEL.PFE_UP_INIT_STD))
        if self.enabled:
            for depth in self.pfe_layers:
                self.pfe_adapters[str(depth)] = RMSPFETokenAdapter(self.base.in_planes, **self.pfe_settings)
                self._pfe_handles.append(self.base.blocks[depth-1].register_forward_hook(self._stage_hook(depth)))

    def architecture(self):
        result = super().architecture()
        result.update(variant=VARIANT,
            residual_scale='fixed image-wide RMS ratio, differentiable FP32; zero output stays zero',
            legacy_pfe_gain_used=False)
        return result
