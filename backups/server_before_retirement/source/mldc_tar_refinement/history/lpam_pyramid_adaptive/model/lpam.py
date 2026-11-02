"""Four free part queries over spatial patches; no region masks or FFT."""
import torch
from torch import nn
import torch.nn.functional as F


class FreeQueryLPAM(nn.Module):
    def __init__(self, input_dim=768, dim=256, layers=(6, 9, 12), queries=4,
                 heads=4, temperature=0.1):
        super().__init__()
        if dim % heads or queries < 2 or temperature <= 0:
            raise ValueError('Invalid LPAM query/attention dimensions')
        self.dim, self.layers, self.queries, self.heads = dim, tuple(layers), queries, heads
        self.temperature = float(temperature)
        self.input_norms = nn.ModuleList(nn.LayerNorm(input_dim) for _ in layers)
        self.projections = nn.ModuleList(nn.Linear(input_dim, dim) for _ in layers)
        self.query = nn.Parameter(torch.empty(queries, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.gate_norm = nn.LayerNorm(dim)
        self.depth_gate = nn.Parameter(torch.zeros(queries, dim))
        self.depth_bias = nn.Parameter(torch.zeros(queries, len(layers)))
        self.mlp = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, patch_layers):
        # [B,L,N,C]. Queries share their identity across depths and images.
        if patch_layers.ndim != 4 or patch_layers.shape[1] != len(self.layers):
            raise ValueError('LPAM expects [batch, selected depths, patches, channels]')
        batch, depth, tokens, _ = patch_layers.shape
        head_dim = self.dim // self.heads
        query = self.query.reshape(self.queries, self.heads, head_dim).permute(1, 0, 2)
        summaries, attentions = [], []
        for i in range(depth):
            projected = self.projections[i](self.input_norms[i](patch_layers[:, i]))
            key = self.key(projected).reshape(batch, tokens, self.heads, head_dim).transpose(1, 2)
            value = self.value(projected).reshape(batch, tokens, self.heads, head_dim).transpose(1, 2)
            # Compute cosine logits in fp32, including under mixed precision.
            with torch.autocast(device_type=patch_layers.device.type, enabled=False):
                q = F.normalize(query.float(), dim=-1, eps=1e-6)
                k = F.normalize(key.float(), dim=-1, eps=1e-6)
                attention = torch.einsum('hrd,bhnd->bhrn', q, k).div(self.temperature).softmax(-1)
                summary = torch.einsum('bhrn,bhnd->bhrd', attention, value.float())
            summaries.append(summary.transpose(1, 2).reshape(batch, self.queries, self.dim))
            attentions.append(attention)
        by_depth = torch.stack(summaries, dim=2)  # [B,R,L,D]
        gate_logits = torch.einsum('brld,rd->brl', self.gate_norm(by_depth), self.depth_gate)
        depth_weights = (gate_logits + self.depth_bias).float().softmax(-1)
        fused = (by_depth * depth_weights.unsqueeze(-1)).sum(2)
        parts = self.output_norm(fused + self.mlp(fused))
        attention_by_depth = torch.stack(attentions, dim=1)  # [B,L,H,R,N]
        with torch.autocast(device_type=patch_layers.device.type, enabled=False):
            maps = torch.einsum('brl,blrn->brn', depth_weights, attention_by_depth.mean(2))
            # Weak overlap penalty on maps, not orthogonality of identity vectors.
            unit_maps = F.normalize(maps.float(), dim=-1, eps=1e-6)
            similarities = torch.bmm(unit_maps, unit_maps.transpose(1, 2))
            indices = torch.triu_indices(self.queries, self.queries, offset=1, device=maps.device)
            overlap_loss = similarities[:, indices[0], indices[1]].square().mean()
        return {'parts': parts, 'attention': maps, 'attention_by_depth': attention_by_depth,
                'depth_weights': depth_weights, 'overlap_loss': overlap_loss}


def unit_local(parts):
    """Equal-weight unit descriptor; each part receives 1/R similarity weight."""
    return F.normalize(parts.float(), dim=-1, eps=1e-12).flatten(1) / parts.shape[1] ** 0.5


def joint_descriptor(global_feature, local_feature, local_weight):
    if not 0 <= local_weight <= 1:
        raise ValueError('Local retrieval weight must be in [0,1]')
    global_feature = F.normalize(global_feature.float(), dim=-1, eps=1e-12)
    local_feature = F.normalize(local_feature.float(), dim=-1, eps=1e-12)
    return torch.cat((global_feature * (1-local_weight)**0.5,
                      local_feature * local_weight**0.5), dim=-1)
