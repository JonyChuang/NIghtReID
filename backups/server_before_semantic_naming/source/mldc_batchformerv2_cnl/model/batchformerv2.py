"""Token-wise batch attention and the first BatchFormerV2 two-stream split.

References: arXiv:2204.01254, Figure 3; author implementation:
https://github.com/zhihou7/BatchFormer/tree/main/batchformer-v2/deit_share

Sequence-first [B,N,C] means sequence=B images, independent batch=N token
positions. CLS and every spatial token are included, as in author defaults.
"""
import torch
from torch import nn


class BatchFormerV2(nn.Module):
    def __init__(self, dim=768, heads=12, dropout=0.5):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim, dropout=dropout,
            activation='relu', batch_first=False, norm_first=False)

    def forward(self, tokens):
        if not self.training:
            return tokens
        if tokens.ndim != 3:
            raise ValueError('BatchFormerV2 requires [images, token positions, channels]')
        return torch.cat((tokens, self.encoder(tokens)), dim=0)
