"""Paper-based MSRFConv/PFEBlock and a constant-width ViT residual adapter.

Source: https://doi.org/10.1007/s44443-026-00927-9, equations 1--18.
Independent implementation; no author code was found. See pfe_provenance.json.
"""
import torch
from torch import nn


class MSRFConv(nn.Module):
    def __init__(self, channels, kernel=3, reduction=32):
        super().__init__()
        if kernel < 1 or kernel % 2 != 1:
            raise ValueError('Use a positive odd receptive-field kernel')
        self.kernel = kernel
        self.generate = nn.Sequential(
            nn.Conv2d(channels, channels * kernel * kernel, kernel,
                      padding=kernel // 2, groups=channels, bias=False),
            nn.BatchNorm2d(channels * kernel * kernel), nn.ReLU())
        hidden = max(8, channels // reduction)
        self.reduce = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False),
                                    nn.BatchNorm2d(hidden), nn.Sigmoid())
        self.height_attention = nn.Conv2d(hidden, channels, 1)
        self.width_attention = nn.Conv2d(hidden, channels, 1)
        self.aggregate = nn.Conv2d(channels, channels, kernel, stride=kernel, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        k = self.kernel
        generated = self.generate(x)
        # Depthwise multiplier k^2 makes this rearrangement element-count exact.
        expanded = generated.reshape(b, c, k, k, h, w).permute(0, 1, 4, 2, 5, 3)
        expanded = expanded.reshape(b, c, h * k, w * k)
        horizontal = expanded.mean(3, keepdim=True)
        vertical = expanded.mean(2, keepdim=True).transpose(2, 3)
        context = self.reduce(torch.cat((horizontal, vertical), 2))
        horizontal, vertical = context.split((h * k, w * k), 2)
        height_weight = self.height_attention(horizontal).sigmoid()
        width_weight = self.width_attention(vertical.transpose(2, 3)).sigmoid()
        return self.aggregate(expanded * height_weight * width_weight)


class PFEBlock(nn.Module):
    def __init__(self, channels, kernel=3, expansion=.5):
        super().__init__()
        hidden = max(1, int(channels * expansion))
        self.msrf = MSRFConv(channels, kernel)
        self.transform = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.ReLU(), nn.Conv2d(hidden, channels, 1, bias=False))

    def increment(self, x):
        return self.transform(self.msrf(x))

    def forward(self, x):
        return x + self.increment(x)


class PFETokenAdapter(nn.Module):
    def __init__(self, input_dim=768, dim=192, kernel=3, expansion=.5, gain=.1):
        super().__init__()
        if not 0 < gain <= 1:
            raise ValueError('PFE residual gain must be in (0,1]')
        self.gain = float(gain)
        self.norm = nn.LayerNorm(input_dim)
        self.down = nn.Linear(input_dim, dim, bias=False)
        self.block = PFEBlock(dim, kernel, expansion)
        self.up = nn.Linear(dim, input_dim, bias=False)
        # Small and nonzero: preserve SSL features while every branch can learn.
        nn.init.normal_(self.up.weight, std=.001)

    def forward(self, patches, grid, diagnostics=False):
        b, n, _ = patches.shape
        if n != grid[0] * grid[1]:
            raise ValueError('Patch sequence does not match its spatial grid')
        projected = self.down(self.norm(patches))
        spatial = projected.transpose(1, 2).reshape(b, -1, *grid)
        # The outer residual preserves the original 768-D tokens, so add only
        # PFEBlock(Z)-Z. Computing the increment directly avoids cancellation.
        increment = self.block.increment(spatial).flatten(2).transpose(1, 2)
        correction = self.gain * self.up(increment)
        enhanced = patches + correction
        stats = {}
        if diagnostics:
            original, changed, delta = patches.detach().float(), enhanced.detach().float(), correction.detach().float()
            stats = dict(before_norm=original.norm(dim=-1).reshape(b, *grid),
                         after_norm=changed.norm(dim=-1).reshape(b, *grid),
                         correction_norm=delta.norm(dim=-1).reshape(b, *grid),
                         correction_ratio=delta.flatten(1).norm(dim=-1) / original.flatten(1).norm(dim=-1).clamp_min(1e-12))
        return enhanced, stats
