"""MLDC readout migrated from the official A3PFN implementation.

Source: https://github.com/zhguoqing/A3PFN/blob/main/model/backbones/vit_pytorch.py
Preserves torch.randn initialization, temperature 0.9, all post-block CLS
vectors, and the weighted sum before the backbone's final LayerNorm.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLDC(nn.Module):
    def __init__(self, depth=12, temperature=0.9):
        super().__init__()
        if depth < 1 or temperature <= 0:
            raise ValueError('MLDC requires a positive depth and temperature')
        self.layer_weights = nn.Parameter(torch.randn(depth))
        self.temperature = float(temperature)

    def normalized_weights(self):
        return F.softmax(self.layer_weights / self.temperature, dim=0)

    def forward(self, features):
        if features.ndim != 3 or features.shape[0] != len(self.layer_weights):
            raise ValueError('MLDC input must be [depth, batch, embedding]')
        weights = self.normalized_weights()
        return torch.sum(features * weights[:, None, None], dim=0)
