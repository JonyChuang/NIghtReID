"""MLDC readout migrated from the official A3PFN implementation.

Source: https://github.com/zhguoqing/A3PFN/blob/main/model/backbones/vit_pytorch.py
Supports the original random initialization and a learnable uniform-start
control, preserving the original random-number consumption in both modes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLDC(nn.Module):
    def __init__(self, depth=12, temperature=0.9, init='random'):
        super().__init__()
        if depth < 1 or temperature <= 0:
            raise ValueError('MLDC requires a positive depth and temperature')
        if init not in ('random', 'uniform'):
            raise ValueError('MLDC init must be random or uniform')
        # Draw in both modes to preserve backbone/classifier initialization.
        logits = torch.randn(depth)
        if init == 'uniform':
            logits.zero_()
        self.layer_weights = nn.Parameter(logits)
        self.temperature = float(temperature)
        self.init = init

    def normalized_weights(self):
        return F.softmax(self.layer_weights / self.temperature, dim=0)

    def forward(self, features):
        if features.ndim != 3 or features.shape[0] != len(self.layer_weights):
            raise ValueError('MLDC input must be [depth, batch, embedding]')
        weights = self.normalized_weights()
        return torch.sum(features * weights[:, None, None], dim=0)
