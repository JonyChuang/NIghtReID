"""ViTMatte residual bottleneck, adapted to a CLS-token ReID backbone.

Source: hustvl/ViTMatte, commit 8cd7ef068380977c3962c4cb733cb1fe7f2241a5,
modeling/backbone/vit.py, ResBottleneckBlock (MIT; see licenses/).
The convolution order, dim//2 width, GELU and zero final norm follow upstream.
Only the Detectron2/fvcore wrappers and token/grid interface are replaced.
"""
import torch
from torch import nn


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels at each spatial position, epsilon 1e-6."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ResidualConvNeck(nn.Module):
    def __init__(self, channels=768):
        super().__init__()
        hidden = channels // 2
        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.norm1 = ChannelLayerNorm(hidden)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1, bias=False)
        self.norm2 = ChannelLayerNorm(hidden)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.norm3 = ChannelLayerNorm(channels)
        for layer in (self.conv1, self.conv2, self.conv3):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.norm3.weight)
        nn.init.zeros_(self.norm3.bias)

    def forward(self, patches):
        residual = self.act1(self.norm1(self.conv1(patches)))
        residual = self.act2(self.norm2(self.conv2(residual)))
        return patches + self.norm3(self.conv3(residual))

    def forward_tokens(self, tokens, grid):
        """Keep CLS unchanged; convolve HxW patch tokens, including rectangular grids."""
        height, width = grid
        batch, length, channels = tokens.shape
        if length != 1 + height * width:
            raise ValueError('Expected exactly one CLS plus H*W patch tokens')
        spatial = tokens[:, 1:].transpose(1, 2).reshape(batch, channels, height, width)
        spatial = self(spatial).flatten(2).transpose(1, 2)
        return torch.cat((tokens[:, :1], spatial), dim=1)


def attach_conv_necks(backbone, layers, placement):
    """Attach after SSL loading so all existing parameter names remain compatible."""
    if placement not in ('pre_attn', 'post_mlp'):
        raise ValueError('Neck placement must be pre_attn or post_mlp')
    for index, block in enumerate(backbone.blocks, 1):
        if index in layers:
            block.conv_neck = ResidualConvNeck(backbone.in_planes)
            block.conv_neck_placement = placement
