"""LocAtViT GAug and PRR, adapted to the existing rectangular TransReID.

Official source: sinahmr/LocAtViT, commit
f99aac4c3d0f417f205faf35742ca25ebcc30ea4, models/locat.py and models/vit.py.
The official MIT license is included in this experiment's LICENSE.locat.
We retain its learned softplus alpha, shifted sigmoid variance, and EPS.
PRR computes only the CLS row: exactly equivalent to computing all rows
and discarding spatial outputs, with substantially less memory.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


def get_eps(grid_size):
    eps = 0.1
    maximum = 0.5 * (max(grid_size) - 1) ** 2
    while maximum / eps ** 2 > 65000 / 2:
        eps *= 2
    return eps


class GaussianAugment(nn.Module):
    def __init__(self, dim, num_heads, grid_size):
        super().__init__()
        self.log_var = nn.Linear(dim // num_heads, 2, bias=True)
        self.log_alpha = nn.Linear(dim // num_heads, 1, bias=True)
        self.grid_size = tuple(grid_size)
        self.eps = get_eps(grid_size)
        self.coef = float(max(grid_size) ** 2)
        self.shift = math.log(self.coef - 1) if self.coef > 2 else 0.
        rows, cols = torch.meshgrid(torch.arange(grid_size[0]), torch.arange(grid_size[1]), indexing='ij')
        coordinates = torch.stack((rows.flatten(), cols.flatten()), -1).float()
        numerator = -0.5 * (coordinates[:, None] - coordinates[None, :]).square()
        # Fixed-resolution 16 x 8 needs neither the official high-resolution
        # fast approximation nor dynamic-grid rescaling/clamping.
        self.register_buffer('numerator', numerator, persistent=False)
        self.capture = False
        self.diagnostics = {}

    def forward(self, q):
        if q.shape[-2] != self.grid_size[0] * self.grid_size[1] + 1:
            raise ValueError('GAug requires one CLS and the configured rectangular patch grid')
        spatial_q = q[..., 1:, :]
        # Force the exponential and softplus paths to FP32 under AMP.
        with torch.autocast(device_type=q.device.type, enabled=False):
            variance = self.coef * torch.sigmoid(F.linear(
                spatial_q.float(), self.log_var.weight.float(), self.log_var.bias.float()) - self.shift)
            alpha = F.softplus(F.linear(spatial_q.float(), self.log_alpha.weight.float(),
                                        self.log_alpha.bias.float()))
            gaussian = torch.exp((self.numerator.float()[None, None] /
                                  (variance[..., None, :] + self.eps)).sum(-1))
            addition = alpha * gaussian
        if self.capture:
            self.diagnostics = {'variance': variance.detach(), 'alpha': alpha.detach()}
        # Leave BOTH CLS row and CLS column unchanged, as in the authors' code.
        return F.pad(addition, (1, 0, 1, 0))


class GaussianAttention(nn.Module):
    def __init__(self, original, dim, grid_size):
        super().__init__()
        self.num_heads, self.scale = original.num_heads, original.scale
        # Reuse SSL-loaded projections without renaming checkpoint keys.
        self.qkv, self.proj = original.qkv, original.proj
        self.attn_drop, self.proj_drop = original.attn_drop, original.proj_drop
        self.gaug = GaussianAugment(dim, self.num_heads, grid_size)
        self.enabled = True
        self.capture = False
        self.diagnostics = {}

    def forward(self, x):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads,
                                  channels // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        content = (q @ k.transpose(-2, -1)) * self.scale
        addition = self.gaug(q) if self.enabled else 0.
        # FP32 softmax is useful with the positive Gaussian logit supplement.
        attention = (content.float() + addition).softmax(-1).to(v.dtype)
        if self.capture:
            self.diagnostics = {'content_attention': content.float().softmax(-1).detach(),
                                'augmented_attention': attention.detach().float()}
        x = (self.attn_drop(attention) @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class PRR(nn.Module):
    """Parameter-free self-attention's CLS output (no residual or new projection)."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.capture = False
        self.attention = None

    def forward(self, x):
        batch, tokens, channels = x.shape
        heads = x.reshape(batch, tokens, self.num_heads, -1).permute(0, 2, 1, 3)
        with torch.autocast(device_type=x.device.type, enabled=False):
            heads = heads.float()
            attention = ((heads[:, :, :1] @ heads.transpose(-2, -1)) * self.scale).softmax(-1)
            output = (attention @ heads).squeeze(2).reshape(batch, channels)
        if self.capture:
            self.attention = attention.detach()
        return output
