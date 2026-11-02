"""GAE, PTF, GMA and PTL adapted from https://github.com/AWangYQ/GLTrans.

The source hard-codes a 21x10 token grid. This version uses the backbone's
actual grid so the same architecture also works with TransReID-SSL's 16x8 grid.
"""
import torch
from torch import nn

from .backbones.vit_pytorch_gltrans_night import Block


def _init_gltrans(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_out')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class PatchAttentionModule(nn.Module):
    """Official PAM: one spatial/channel gate for each selected ViT layer."""

    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(1, dim // reduction)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, dim, 1, bias=False),
            nn.BatchNorm2d(dim), nn.Sigmoid(),
        )

    def forward(self, tokens, grid_size):
        height, width = grid_size
        batch, count, dim = tokens.shape
        if count != height * width:
            raise ValueError('Patch token count does not match the ViT grid')
        feature_map = tokens.transpose(1, 2).reshape(batch, dim, height, width)
        return feature_map + feature_map * self.gate(feature_map)


class PatchTokenFusion(nn.Module):
    """PTF from the official minimodule: 1x1/GELU/BN then 3x3 convolution."""

    def __init__(self, dim, num_layers):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(num_layers * dim, dim, kernel_size=1),
            nn.GELU(), nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, maps):
        return self.fusion(torch.cat(maps, dim=1))


class GlobalGuidedAttention(nn.Module):
    """Official GMA: sigmoid gate per head and patch, guided by GAE output."""

    def __init__(self, dim, heads):
        super().__init__()
        if dim % heads:
            raise ValueError('Embedding dimension must be divisible by head count')
        self.heads = heads
        self.global_linear = nn.Linear(dim, dim)
        self.local_linear = nn.Linear(dim, dim)

    def forward(self, global_feature, local_tokens):
        batch, count, dim = local_tokens.shape
        width = dim // self.heads
        global_key = self.global_linear(global_feature[:, None]).reshape(
            batch, 1, self.heads, width).transpose(1, 2)
        local_query = self.local_linear(local_tokens).reshape(
            batch, count, self.heads, width).transpose(1, 2)
        gate = torch.sigmoid(local_query @ global_key.transpose(-2, -1))
        values = local_tokens.reshape(batch, count, self.heads, width).transpose(1, 2)
        return local_tokens + (values * gate).transpose(1, 2).reshape(batch, count, dim)


class PartTransformerLayers(nn.Module):
    """PTL: three horizontal parts, each with a learned CLS and two ViT blocks."""

    def __init__(self, dim, heads, parts=3):
        super().__init__()
        self.parts = parts
        self.part_cls = nn.Parameter(torch.zeros(parts, 1, 1, dim))
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                Block(dim=dim, num_heads=heads, mlp_ratio=4, qkv_bias=True,
                      drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm)
                for _ in range(parts)
            ]) for _ in range(2)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens, class_tokens, grid_size):
        height, width = grid_size
        batch, count, dim = tokens.shape
        if count != height * width or height < self.parts:
            raise ValueError('PTL needs a valid patch grid with at least one row per part')
        feature_map = tokens.transpose(1, 2).reshape(batch, dim, height, width)
        rows, extra = divmod(height, self.parts)
        output = []
        start = 0
        for part in range(self.parts):
            stop = start + rows + (extra if part == self.parts - 1 else 0)
            patch_part = feature_map[:, :, start:stop].flatten(2).transpose(1, 2)
            sequence = torch.cat((self.part_cls[part].expand(batch, -1, -1),
                                  class_tokens, patch_part), dim=1)
            for layer in self.blocks:
                sequence = layer[part](sequence)
            output.append(self.norm(sequence)[:, 0])
            start = stop
        return torch.cat(output, dim=1)


class GLTransModules(nn.Module):
    """Complete official global/local path after the three selected ViT blocks."""

    def __init__(self, dim, heads, parts=3, num_layers=3):
        super().__init__()
        if num_layers != 3:
            raise ValueError('The official GLTrans path uses exactly three layers')
        self.pam = nn.ModuleList([PatchAttentionModule(dim) for _ in range(num_layers)])
        self.gae = nn.Linear(num_layers * dim, dim)
        self.ptf = PatchTokenFusion(dim, num_layers)
        self.gma = GlobalGuidedAttention(dim, heads)
        self.ptl = PartTransformerLayers(dim, heads, parts)
        self.apply(_init_gltrans)

    def forward(self, patch_tokens, class_tokens, grid_size):
        if len(patch_tokens) != len(self.pam) or class_tokens.shape[1] != len(self.pam):
            raise ValueError('GAE/PTF need matching class and patch tokens')
        batch = class_tokens.shape[0]
        global_feature = self.gae(class_tokens.reshape(batch, -1))
        patch_maps = [module(tokens, grid_size)
                      for module, tokens in zip(self.pam, patch_tokens)]
        fused_map = self.ptf(patch_maps)
        fused_tokens = fused_map.flatten(2).transpose(1, 2)
        guided_tokens = self.gma(global_feature, fused_tokens)
        local_feature = self.ptl(guided_tokens, class_tokens, grid_size)
        return global_feature, local_feature
