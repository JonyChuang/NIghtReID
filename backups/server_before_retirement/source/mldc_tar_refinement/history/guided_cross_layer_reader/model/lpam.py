"""Competitive free queries over a 3/6/9 patch pyramid; no region masks or FFT."""
import torch
from torch import nn
import torch.nn.functional as F


class FreeQueryLPAM(nn.Module):
    def __init__(self, input_dim=768, dim=256, layers=(3, 6, 9), queries=4,
                 heads=4, temperature=0.1):
        super().__init__()
        if dim % heads or queries < 2 or temperature <= 0:
            raise ValueError('Invalid LPAM query/attention dimensions')
        self.dim, self.layers, self.queries, self.heads = dim, tuple(layers), queries, heads
        self.temperature = float(temperature)
        self.input_norms = nn.ModuleList(nn.LayerNorm(input_dim) for _ in layers)
        self.projections = nn.ModuleList(nn.Linear(input_dim, dim) for _ in layers)
        self.pyramid_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in layers)
        self.top_down = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in layers[:-1])
        for projection in self.top_down:
            nn.init.eye_(projection.weight)
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
        if patch_layers.ndim != 4 or patch_layers.shape[1] != len(self.layers):
            raise ValueError('LPAM expects [batch, selected depths, patches, channels]')
        batch, depth, tokens, _ = patch_layers.shape
        head_dim = self.dim // self.heads
        # All selected ViT blocks have the same spatial grid. Fuse corresponding
        # patch positions from deep to shallow; no spatial resizing or hard bands.
        projected = [projection(norm(patch_layers[:, i])) for i, (norm, projection)
                     in enumerate(zip(self.input_norms, self.projections))]
        pyramid = [None]*depth
        pyramid[-1] = self.pyramid_norms[-1](projected[-1])
        for i in range(depth-2, -1, -1):
            pyramid[i] = self.pyramid_norms[i](projected[i]+self.top_down[i](pyramid[i+1]))
        query = self.query.reshape(self.queries, self.heads, head_dim).permute(1, 0, 2)
        summaries, attentions = [], []
        for features in pyramid:
            key = self.key(features).reshape(batch, tokens, self.heads, head_dim).transpose(1, 2)
            value = self.value(features).reshape(batch, tokens, self.heads, head_dim).transpose(1, 2)
            with torch.autocast(device_type=patch_layers.device.type, enabled=False):
                q = F.normalize(query.float(), dim=-1, eps=1e-6)
                k = F.normalize(key.float(), dim=-1, eps=1e-6)
                logits = torch.einsum('hrd,bhnd->bhrn', q, k)/self.temperature
                # Each patch first competes across the four queries. Normalize
                # over patches afterwards to obtain each query's weighted mean.
                assignment = logits.softmax(dim=2)
                attention = assignment/assignment.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                summary = torch.einsum('bhrn,bhnd->bhrd', attention, value.float())
            summaries.append(summary.transpose(1, 2).reshape(batch, self.queries, self.dim))
            attentions.append(attention)
        by_depth = torch.stack(summaries, dim=2)
        gate_logits = torch.einsum('brld,rd->brl', self.gate_norm(by_depth), self.depth_gate)
        depth_weights = (gate_logits+self.depth_bias).float().softmax(-1)
        fused = (by_depth*depth_weights.unsqueeze(-1)).sum(2)
        parts = self.output_norm(fused+self.mlp(fused))
        attention_by_depth = torch.stack(attentions, dim=1)
        with torch.autocast(device_type=patch_layers.device.type, enabled=False):
            maps = torch.einsum('brl,blrn->brn', depth_weights, attention_by_depth.mean(2))
            unit_maps = F.normalize(maps.float(), dim=-1, eps=1e-6)
            similarities = torch.bmm(unit_maps, unit_maps.transpose(1, 2))
            indices = torch.triu_indices(self.queries, self.queries, offset=1, device=maps.device)
            overlap_loss = similarities[:, indices[0], indices[1]].square().mean()
        return {'parts': parts, 'attention': maps, 'attention_by_depth': attention_by_depth,
                'depth_weights': depth_weights, 'overlap_loss': overlap_loss}


class AdaptiveFeatureFusion(nn.Module):
    """Predict per-image weights for one global and four local unit vectors."""
    def __init__(self, global_dim=768, local_dim=256, queries=4, hidden=128,
                 initial_local_weight=0.2):
        super().__init__()
        if not 0 < initial_local_weight < 1:
            raise ValueError('Initial local weight must be strictly between zero and one')
        self.queries, self.hidden = queries, hidden
        dimension = global_dim+queries*local_dim
        self.gate = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, hidden),
                                  nn.GELU(), nn.Linear(hidden, queries+1))
        nn.init.zeros_(self.gate[-1].weight)
        prior = torch.tensor([1-initial_local_weight]+[initial_local_weight/queries]*queries)
        with torch.no_grad():
            self.gate[-1].bias.copy_(prior.log())

    def forward(self, global_feature, part_features):
        with torch.autocast(device_type=global_feature.device.type, enabled=False):
            global_unit = F.normalize(global_feature.float(), dim=-1, eps=1e-12)
            part_units = F.normalize(part_features.float(), dim=-1, eps=1e-12)
            # The new fusion objective must not change the global head's
            # original supervision through either the gate input or descriptor.
            gate_input = torch.cat((global_unit.detach(), part_units.flatten(1)), dim=-1)
            log_weights = self.gate(gate_input).float().log_softmax(-1)
            weights = log_weights.exp()
            # exp(log_w/2) avoids the singular derivative of sqrt at underflow.
            root_weights = (0.5*log_weights).exp()
            global_segment = global_unit*root_weights[:, :1]
            local_segment = (part_units*root_weights[:, 1:, None]).flatten(1)
            retrieval = torch.cat((global_segment, local_segment), dim=-1)
            fusion_local_metric = torch.cat((global_unit.detach()*root_weights[:, :1],
                                              local_segment), dim=-1)
        return {'retrieval': retrieval, 'fusion_local_metric': fusion_local_metric,
                'fusion_weights': weights}


def unit_local(parts):
    """Equal-weight unit descriptor; each part receives 1/R similarity weight."""
    return F.normalize(parts.float(), dim=-1, eps=1e-12).flatten(1)/parts.shape[1]**0.5


def joint_descriptor(global_feature, local_feature, local_weight):
    if not 0 <= local_weight <= 1:
        raise ValueError('Local retrieval weight must be in [0,1]')
    global_feature = F.normalize(global_feature.float(), dim=-1, eps=1e-12)
    local_feature = F.normalize(local_feature.float(), dim=-1, eps=1e-12)
    return torch.cat((global_feature*(1-local_weight)**0.5,
                      local_feature*local_weight**0.5), dim=-1)
