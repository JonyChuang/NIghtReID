"""TAR refinement adapted from VectorYangYiStar/TAR model.py.

Source revision: 731e4b33741b66b40a1b5f01a341648adc834311.
TRM/CATS/GRM follow the public implementation; computations use FP32.
The released GET training uses KL, without the computed perturbation loss.
Fix the source TRM's discarded scatter return so candidates are populated.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TokenAwareRefinement(nn.Module):
    def __init__(self, dim=768, bottleneck=256, top_ratio=0.5, norm_alpha=2.0,
                 temperature=0.8, kl_beta=0.05):
        super().__init__()
        if not 0 < top_ratio <= 1 or temperature <= 0:
            raise ValueError('Require 0 < TAR_TOP_RATIO <= 1 and positive temperature')
        self.dim, self.bottleneck = dim, bottleneck
        self.top_ratio, self.norm_alpha = float(top_ratio), float(norm_alpha)
        self.temperature, self.kl_beta = float(temperature), float(kl_beta)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, bottleneck), nn.GELU(), nn.Linear(bottleneck, dim))
        self.gate = nn.Linear(2 * dim, 1)
        self.gate_alpha = nn.Parameter(torch.tensor(1.2))
        self.gate_beta = nn.Parameter(torch.tensor(-0.1))
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, cls, patches):
        with torch.autocast(device_type=patches.device.type, enabled=False):
            cls, patches = cls.float(), patches.float()
            norms = patches.norm(dim=-1)
            sigma = norms.std(dim=1, keepdim=True, unbiased=patches.shape[1] > 1) + 1e-6
            zscore = (norms - norms.mean(dim=1, keepdim=True)) / sigma
            weights = torch.sigmoid(-self.norm_alpha * zscore)
            weighted = patches * weights.unsqueeze(-1)
            count = max(1, int(self.top_ratio * patches.shape[1]))
            indices = weights.topk(count, dim=1, largest=True).indices
            mask = torch.zeros_like(weights, dtype=torch.bool).scatter(1, indices, True)
            scores = (self.query(cls).unsqueeze(1) * self.key(weighted)).sum(-1) / math.sqrt(self.dim)
            scores = torch.where(mask, scores / self.temperature, torch.full_like(scores, -1e9))
            scores = scores - scores.max(dim=1, keepdim=True).values
            attention = F.softmax(scores, dim=1) * mask.float()
            attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-12)
            entropy = -(attention * attention.clamp_min(1e-12).log()).sum(dim=1)
            kl = (mask.sum(dim=1).float().log() - entropy).mean()
            detail = self.norm(torch.einsum('bn,bnd->bd', attention, self.value(weighted)))
            detail = detail + 0.5 * self.mlp(detail)
            gamma = self.gate_alpha * torch.sigmoid(self.gate(torch.cat([cls, detail], dim=1))) + self.gate_beta
            delta = gamma * detail
            return {'refined_cls': cls + delta, 'tar_delta': delta, 'tar_detail': detail,
                    'tar_gate': gamma, 'tar_attention': attention, 'tar_mask': mask,
                    'tar_weights': weights, 'tar_patch_norms': norms,
                    'tar_kl': kl, 'tar_kl_weighted': self.kl_beta * kl}
