"""MLDC readout migrated from the official A3PFN implementation.

Source: https://github.com/zhguoqing/A3PFN/blob/main/model/backbones/vit_pytorch.py
Supports the original random initialization and a learnable uniform-start
control, preserving the original random-number consumption in both modes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLDC(nn.Module):
    def __init__(self, depth=12, temperature=0.9, init='random', deep_mass=0.75):
        super().__init__()
        if depth < 1 or temperature <= 0:
            raise ValueError('MLDC requires a positive depth and temperature')
        if init not in ('random', 'uniform', 'deep_prior'):
            raise ValueError('MLDC init must be random, uniform, or deep_prior')
        # Draw in both modes to preserve backbone/classifier initialization.
        logits = torch.randn(depth)
        if init == 'uniform':
            logits.zero_()
        elif init == 'deep_prior':
            if depth <= 4 or not 0 < deep_mass < 1:
                raise ValueError('Deep prior requires depth > 4 and 0 < deep_mass < 1')
            probabilities = torch.full_like(logits, (1-deep_mass)/(depth-4))
            probabilities[-4:] = deep_mass/4
            logits.copy_(temperature * probabilities.log())
            logits.sub_(logits.mean())
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
