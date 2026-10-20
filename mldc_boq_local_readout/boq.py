"""BoQ-style aggregation adapted to multiple ViT patch depths.

Adapted from the MIT-licensed BoQBlock/BoQ reference implementation:
https://github.com/amaralibey/Bag-of-Queries#open-source-code
Copyright (c) 2024 Amar Ali-bey. See BOQ_LICENSE.txt.
"""
import torch
from torch import nn


class BoQBlock(nn.Module):
    def __init__(self, dim, num_queries=4, heads=4):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=4*dim,
                                                  dropout=0.0, batch_first=True)
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim))
        self.query_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.query_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, patches):
        x = self.encoder(patches)
        q = self.queries.expand(x.shape[0], -1, -1)
        q = self.query_norm(q + self.query_attention(q, q, q, need_weights=False)[0])
        # The query is a reading operator. It is not added to the output.
        output, attention = self.cross_attention(q, x, x, need_weights=True,
                                                 average_attn_weights=False)
        return self.output_norm(output), attention


class BoQPatchReader(nn.Module):
    def __init__(self, input_dim=768, dim=192, heads=4, layers=(4, 6, 8),
                 queries=4, row_dim=None):
        super().__init__()
        self.layers = tuple(layers)
        if (not self.layers or tuple(sorted(set(self.layers))) != self.layers
                or any(type(i) is not int or not 1 <= i <= 12 for i in self.layers)):
            raise ValueError('Patch layers must be increasing unique integers in [1,12]')
        if dim <= 0 or heads <= 0 or dim % heads or queries < 1:
            raise ValueError('Invalid BoQ attention dimensions')
        row_dim = input_dim // dim if row_dim is None else row_dim
        if row_dim < 1 or dim * row_dim != input_dim:
            raise ValueError('BoQ channel dimension times row dimension must equal the local descriptor dimension')
        self.dim, self.heads, self.queries, self.row_dim = dim, heads, queries, row_dim
        self.mode = 'boq'
        self.input_norms = nn.ModuleList(nn.LayerNorm(input_dim) for _ in self.layers)
        self.projections = nn.ModuleList(nn.Linear(input_dim, dim) for _ in self.layers)
        self.projected_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in self.layers)
        # Each selected backbone depth is a separate input to a BoQ block.
        # Unlike canonical BoQ, these blocks do not form a cascade over one map.
        self.blocks = nn.ModuleList(BoQBlock(dim, queries, heads) for _ in self.layers)
        self.slot_projection = nn.Linear(len(self.layers) * queries, row_dim)
        self.descriptor_norm = nn.LayerNorm(input_dim)

    def forward(self, patches):
        if len(patches) != len(self.layers):
            raise ValueError('One post-block patch tensor is required per selected depth')
        summaries, maps = [], []
        for norm, projection, projected_norm, block, patch in zip(
                self.input_norms, self.projections, self.projected_norms, self.blocks, patches):
            if patch.ndim != 3:
                raise ValueError('Patch tensors must have shape [batch, patches, channels]')
            x = projected_norm(projection(norm(patch)))
            summary, attention = block(x)
            summaries.append(summary)
            maps.append(attention)
        # Preserve every query/depth output until a learned slot projection.
        bag = torch.cat(summaries, dim=1)  # B, depths*queries, channels
        local = self.slot_projection(bag.transpose(1, 2)).flatten(1)
        local = self.descriptor_norm(local)
        attention = torch.stack(maps, dim=2)  # B, heads, depths, queries, patches
        return local, attention
