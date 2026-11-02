"""Scheme B: feature-space adaptation of FEN/SAFA, not the original RGB module."""
import torch
from torch import nn
from torch.nn import functional as F


def conv(in_channels, out_channels):
    # Replication preserves spatially constant inputs without padding edges.
    return nn.Conv2d(in_channels, out_channels, 3, padding=1,
                     padding_mode='replicate', bias=False)


class PatchFEN(nn.Module):
    """Shared seven-convolution extractor with the original concatenation paths."""
    def __init__(self, channels, hidden):
        super().__init__()
        self.e_conv1 = conv(channels, hidden)
        self.e_conv2 = conv(hidden, hidden)
        self.e_conv3 = conv(hidden, hidden)
        self.e_conv4 = conv(hidden, hidden)
        self.e_conv5 = conv(hidden * 2, hidden)
        self.e_conv6 = conv(hidden * 2, hidden)
        self.e_conv7 = conv(hidden * 2, channels)

    def forward(self, x):
        x1 = F.relu(self.e_conv1(x))
        x2 = F.relu(self.e_conv2(x1))
        x3 = F.relu(self.e_conv3(x2))
        x4 = F.relu(self.e_conv4(x3))
        x5 = F.relu(self.e_conv5(torch.cat((x3, x4), 1)))
        x6 = F.relu(self.e_conv6(torch.cat((x2, x5), 1)))
        return self.e_conv7(torch.cat((x1, x6), 1))


class PatchSAFA(nn.Module):
    """Eight-group, two-scale attention at the coarse grid, then upsampling."""
    def __init__(self, channels, heads=8):
        super().__init__()
        if channels % heads:
            raise ValueError('Patch dimension must be divisible by eight')
        self.heads = heads
        self.query = nn.Conv2d(channels, channels, 1, bias=False)
        self.key = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, fine, middle, coarse_size):
        query = self.query(F.adaptive_avg_pool2d(fine, coarse_size))
        key = self.key(F.adaptive_avg_pool2d(middle, coarse_size))
        b, c, h, w = query.shape
        sources = torch.stack((query, key), 1).reshape(b, 2, self.heads, c // self.heads, h, w)
        # Softmax selects source scales, as in SAFA; this is not spatial attention.
        with torch.autocast(device_type=query.device.type, enabled=False):
            sources = sources.float()
            logits = (sources * sources[:, 1:2]).sum(3, keepdim=True) / (c // self.heads) ** .5
            weights = logits.softmax(1)
            aggregated = (sources * weights).sum(1).reshape(b, c, h, w)
        return fine + F.interpolate(aggregated.to(fine.dtype), fine.shape[-2:], mode='bilinear', align_corners=False)


class PatchFeatEnHancer(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.fen = PatchFEN(channels, hidden)
        self.safa = PatchSAFA(channels)
        self.fuse = conv(channels * 2, hidden)
        self.output = conv(hidden, channels)
        # Nonzero initialization lets the complete enhancement path learn at step 1.
        nn.init.normal_(self.output.weight, std=.001)

    def forward(self, x):
        if min(x.shape[-2:]) < 4:
            raise ValueError('Feature pyramid needs at least a 4x4 patch grid')
        fine = self.fen(x)
        middle = self.fen(F.avg_pool2d(x, 2))
        coarse = self.fen(F.avg_pool2d(x, 4))
        aggregated = self.safa(fine, middle, coarse.shape[-2:])
        coarse = F.interpolate(coarse, x.shape[-2:], mode='bilinear', align_corners=False)
        correction = self.output(F.relu(self.fuse(torch.cat((aggregated, coarse), 1))))
        return x + correction


class PatchCompensation(nn.Module):
    """CLS-conditioned directional attention over centered, enhanced patch values."""
    def __init__(self, input_dim, dim, hidden, layers, gain):
        super().__init__()
        self.layers = tuple(layers)
        if not self.layers or sorted(set(self.layers)) != list(self.layers) or not all(1 <= x <= 12 for x in self.layers):
            raise ValueError('Patch layers must be sorted, unique depths in 1..12')
        if not 0 < gain <= 1:
            raise ValueError('Use a positive, bounded fixed compensation gain')
        self.gain = float(gain)
        self.projections = nn.ModuleList([nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim, bias=False)) for _ in layers])
        self.enhancer = PatchFeatEnHancer(dim, hidden)
        self.token_norm = nn.LayerNorm(dim)
        self.query = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim, bias=False))
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, input_dim, bias=False)
        self.logit_scale = 8.

    def forward(self, patches, global_feature, grid, diagnostics=False):
        if patches.ndim != 4 or patches.shape[0] != len(self.layers) or patches.shape[2] != grid[0] * grid[1]:
            raise ValueError('Expected [selected depths, batch, grid tokens, embedding]')
        query = self.query(global_feature)
        reads, maps, norms, changes, source_norms, delta_norms = [], [], [], [], [], []
        for projection, source in zip(self.projections, patches):
            tokens = projection(source)
            b, n, d = tokens.shape
            spatial = tokens.transpose(1, 2).reshape(b, d, *grid)
            enhanced = self.enhancer(spatial)
            normalized = self.token_norm(enhanced.flatten(2).transpose(1, 2))
            keys = self.key(normalized)
            values = self.value(normalized)
            with torch.autocast(device_type=tokens.device.type, enabled=False):
                logits = torch.einsum('bd,bnd->bn', F.normalize(query.float(), dim=-1), F.normalize(keys.float(), dim=-1)) * self.logit_scale
                attention = logits.softmax(-1)
                centered_values = values.float() - values.float().mean(1, keepdim=True)
                # Equivalent to attention-weighted centered values; explicit
                # centering of weights makes a uniform read exactly zero.
                reads.append(torch.einsum('bn,bnd->bd', attention - 1. / n, centered_values))
            if diagnostics:
                maps.append(attention.reshape(b, *grid))
                norms.append(source.float().norm(dim=-1).reshape(b, *grid))
                changes.append((enhanced.float()-spatial.float()).norm(dim=1))
                source_norms.append(spatial.float().flatten(1).norm(dim=-1))
                delta_norms.append((enhanced.float()-spatial.float()).flatten(1).norm(dim=-1))
        local = self.output(torch.stack(reads).mean(0).to(query.dtype))
        # Fix the compensation energy scale; attention/value direction remains learned.
        with torch.autocast(device_type=local.device.type, enabled=False):
            local = F.normalize(local.float(), dim=-1, eps=1e-6) * global_feature.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        result = dict(local_feature=local, compensated=global_feature + self.gain * local)
        if diagnostics:
            result.update(patch_attention=torch.stack(maps, 1), patch_norm=torch.stack(norms, 1),
                          patch_change=torch.stack(changes, 1), patch_source_norm=torch.stack(source_norms, 1),
                          patch_delta_norm=torch.stack(delta_norms, 1))
        return result
