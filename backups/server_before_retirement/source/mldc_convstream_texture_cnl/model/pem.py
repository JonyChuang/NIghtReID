"""PEM-style sparse interaction of the existing CLS and spatial tokens."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SparseTokenInteraction(nn.Module):
    """Joint token update; four scalar gates are the only new model parameters.

    This is a ReID adaptation, not a reproduction of class-specific MuP-VSS
    prototypes: use the image's existing CLS and the existing spatial tokens.
    """
    def __init__(self, dim=768, threshold=.5, temperature=1., gate_init=0.):
        super().__init__()
        if dim < 1 or not -1 <= threshold <= 1 or temperature <= 0 or not math.isfinite(gate_init):
            raise ValueError('Invalid sparse token interaction settings')
        self.dim = int(dim)
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.last_statistics = {}

    def interaction(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1] < 2 or tokens.shape[-1] != self.dim:
            raise ValueError('Expected CLS plus spatial tokens, shape [B,N,D]')
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            # Parameter-free normalization gives a common cosine threshold.
            normalized = F.layer_norm(tokens.float(), (self.dim,), eps=1e-6)
            normalized = F.normalize(normalized, dim=-1, eps=1e-12)
            affinity = normalized @ normalized.transpose(-1, -2)
            diagonal = torch.eye(tokens.shape[1], dtype=torch.bool, device=tokens.device)[None]
            keep = (torch.ones_like(affinity, dtype=torch.bool) if self.threshold <= -1.
                    else (affinity > self.threshold) | diagonal)
            logits = (affinity / self.temperature).masked_fill(~keep, float('-inf'))
            weights = logits.softmax(-1)
            message = weights @ tokens.float()
        return message.to(tokens.dtype), affinity, keep, weights

    def forward(self, tokens):
        message, affinity, keep, weights = self.interaction(tokens)
        update = self.gate.to(tokens.dtype) * message
        result = tokens + update
        with torch.no_grad():
            b, n, _ = tokens.shape
            diagonal = torch.eye(n, dtype=torch.bool, device=tokens.device)[None]
            self.last_statistics = dict(
                kept_fraction=((keep & ~diagonal).float().sum() / (b*n*(n-1))).detach(),
                cls_to_patch_kept_fraction=keep[:, 0, 1:].float().mean().detach(),
                patch_to_cls_kept_fraction=keep[:, 1:, 0].float().mean().detach(),
                off_diagonal_cosine_mean=affinity.masked_select(~diagonal.expand(b, -1, -1)).mean().detach(),
                cls_update_norm_ratio=(update[:, 0].float().norm(dim=-1).mean() /
                    tokens[:, 0].float().norm(dim=-1).mean().clamp_min(1e-12)).detach(),
                patch_update_norm_ratio=(update[:, 1:].float().norm(dim=-1).mean() /
                    tokens[:, 1:].float().norm(dim=-1).mean().clamp_min(1e-12)).detach(),
                attention_entropy=(-(weights*weights.clamp_min(1e-30).log()).sum(-1).mean()).detach())
        return result
