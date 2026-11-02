"""CVPR 2022 BatchFormer on the image-batch dimension.

Reference: https://github.com/zhihou7/BatchFormer (README, PyTorch Code).
Input [N,C] becomes [sequence=N,batch=1,C], without positional encoding.
"""
from torch import nn


class BatchFormer(nn.Module):
    def __init__(self, dim=768, heads=4, dropout=0.5):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim, dropout=dropout,
            activation='relu', batch_first=False, norm_first=False)

    def forward(self, features):
        if not self.training:
            return features
        return self.encoder(features.unsqueeze(1)).squeeze(1)
