"""MLDC readout migrated from the official A3PFN implementation.

Source: https://github.com/zhguoqing/A3PFN/blob/main/model/backbones/vit_pytorch.py
Supports the original random initialization and a learnable uniform-start
control, preserving the original random-number consumption in both modes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLDC(nn.Module):
    def __init__(self, depth=12, temperature=0.9, init='random', deep_mass=0.75,
                 selected_layers=None):
        super().__init__()
        if depth < 1 or temperature <= 0:
            raise ValueError('MLDC requires a positive depth and temperature')
        if init not in ('random', 'uniform', 'deep_prior'):
            raise ValueError('MLDC init must be random, uniform, or deep_prior')
        layers = tuple(range(1, depth + 1)) if selected_layers is None else tuple(selected_layers)
        if (not layers or any(type(layer) is not int or not 1 <= layer <= depth for layer in layers)
                or tuple(sorted(set(layers))) != layers):
            raise ValueError('MLDC layers must be unique increasing 1-based layer numbers')
        if init == 'deep_prior' and len(layers) != depth:
            raise ValueError('Deep-prior initialization requires all layers')
        self.depth = depth
        self.selected_layers = layers
        self.register_buffer('selected_indices', torch.tensor([layer - 1 for layer in layers]),
                             persistent=False)
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
        # Always consume depth random draws; only selected logits are trainable.
        self.layer_weights = nn.Parameter(logits.index_select(0, self.selected_indices))
        self.temperature = float(temperature)
        self.init = init

    def normalized_weights(self):
        weights = F.softmax(self.layer_weights / self.temperature, dim=0)
        return weights.new_zeros(self.depth).index_copy(0, self.selected_indices, weights)

    def forward(self, features):
        if features.ndim != 3 or features.shape[0] != self.depth:
            raise ValueError('MLDC input must be [depth, batch, embedding]')
        weights = F.softmax(self.layer_weights / self.temperature, dim=0)
        selected = features.index_select(0, self.selected_indices)
        return torch.sum(selected * weights[:, None, None], dim=0)
