"""MLDC-conditioned reads of unmodified per-layer patch sequences."""
import torch
from torch import nn
import torch.nn.functional as F


class MLDCGuidedReadout(nn.Module):
    def __init__(self, input_dim=768, dim=256, layers=(3, 6, 9), heads=4):
        super().__init__()
        if dim % heads:
            raise ValueError('Readout dimension must be divisible by head count')
        self.dim, self.layers, self.heads = dim, tuple(layers), heads
        self.branches = len(self.layers)
        self.query_norm = nn.LayerNorm(input_dim)
        self.patch_norms = nn.ModuleList(nn.LayerNorm(input_dim) for _ in layers)
        self.attentions = nn.ModuleList(
            nn.MultiheadAttention(dim, heads, kdim=input_dim, vdim=input_dim,
                                  dropout=0.0, batch_first=True) for _ in layers)
        self.query_projections = nn.ModuleList(nn.Linear(input_dim, dim) for _ in layers)
        self.read_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in layers)
        self.mlps = nn.ModuleList(nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(),
                                                nn.Linear(2*dim, dim)) for _ in layers)
        self.output_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in layers)

    def forward(self, global_feature, patch_layers):
        if patch_layers.ndim != 4 or patch_layers.shape[1] != self.branches:
            raise ValueError('Readout expects [batch, selected layers, patches, channels]')
        query_source = self.query_norm(global_feature)
        reads, attention_heads = [], []
        for index in range(self.branches):
            query = self.query_projections[index](query_source).unsqueeze(1)
            patches = self.patch_norms[index](patch_layers[:, index])
            read, weights = self.attentions[index](
                query, patches, patches, need_weights=True, average_attn_weights=False)
            # Only patch values enter the output. Do not add the global query
            # as a residual, or fuse patch sequences across depths beforehand.
            read = self.read_norms[index](read[:, 0])
            reads.append(self.output_norms[index](read+self.mlps[index](read)))
            attention_heads.append(weights.squeeze(2))
        attention_heads = torch.stack(attention_heads, dim=1)
        return {'raw_reads': torch.stack(reads, dim=1),
                'attention': attention_heads.mean(2),
                'attention_heads': attention_heads}


class AdaptiveFeatureFusion(nn.Module):
    """Learn image-dependent weights for MLDC and each layer readout."""
    def __init__(self, global_dim=768, local_dim=256, branches=3, hidden=128,
                 initial_local_weight=0.2):
        super().__init__()
        if not 0 < initial_local_weight < 1:
            raise ValueError('Initial local weight must be strictly between zero and one')
        self.branches, self.hidden = branches, hidden
        dimension = global_dim+branches*local_dim
        self.gate = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, hidden),
                                  nn.GELU(), nn.Linear(hidden, branches+1))
        nn.init.zeros_(self.gate[-1].weight)
        prior = torch.tensor([1-initial_local_weight]+[initial_local_weight/branches]*branches)
        with torch.no_grad():
            self.gate[-1].bias.copy_(prior.log())

    def forward(self, global_feature, read_features):
        with torch.autocast(device_type=global_feature.device.type, enabled=False):
            global_unit = F.normalize(global_feature.float(), dim=-1, eps=1e-12)
            read_units = F.normalize(read_features.float(), dim=-1, eps=1e-12)
            gate_input = torch.cat((global_unit, read_units.flatten(1)), dim=-1)
            log_weights = self.gate(gate_input).float().log_softmax(-1)
            root_weights = (0.5*log_weights).exp()
            retrieval = torch.cat((global_unit*root_weights[:, :1],
                                   (read_units*root_weights[:, 1:, None]).flatten(1)), dim=-1)
        return {'retrieval': retrieval, 'fusion_weights': log_weights.exp()}
