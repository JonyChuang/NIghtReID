"""Patch-only identity readout and bounded GRM fusion; no global query shortcut."""
import math
import torch
from torch import nn


class GateRefinement(nn.Module):
    def __init__(self, hidden_dim=192, input_dim=768, mlp_hidden=256,
                 max_gate=0.1, initial_gate=0.01):
        super().__init__()
        if not (math.isfinite(max_gate) and math.isfinite(initial_gate)
                and 0 < initial_gate < max_gate):
            raise ValueError('Require 0 < initial gate < maximum gate')
        self.max_gate, self.initial_gate = float(max_gate), float(initial_gate)
        self.mlp_hidden = int(mlp_hidden)
        self.projection = nn.Linear(hidden_dim, input_dim, bias=False)
        self.local_norm = nn.LayerNorm(input_dim)
        self.refinement = nn.Sequential(nn.Linear(input_dim, mlp_hidden), nn.GELU(),
                                        nn.Linear(mlp_hidden, input_dim))
        self.gate_global_norm = nn.LayerNorm(input_dim)
        self.gate_local_norm = nn.LayerNorm(input_dim)
        self.gate = nn.Linear(input_dim * 2, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(initial_gate / (max_gate - initial_gate)))

    def refine(self, pooled):
        local = self.local_norm(self.projection(pooled))
        return local + 0.5 * self.refinement(local)

    def fuse(self, anchor, local, enabled):
        if not enabled:
            return torch.zeros_like(anchor), anchor.new_zeros(anchor.shape[0], 1)
        inputs = torch.cat([self.gate_global_norm(anchor), self.gate_local_norm(local)], -1)
        logits = self.gate(inputs)
        with torch.autocast(device_type=anchor.device.type, enabled=False):
            gate = self.max_gate * torch.sigmoid(logits.float())
            # Only the fusion scale receives the anchor norm; the independently
            # supervised local feature never consumes any CLS information.
            scale = anchor.float().norm(dim=-1, keepdim=True).detach()
            direction = local.float() / (local.float().norm(dim=-1, keepdim=True) + 1e-6)
            delta = gate * scale * direction
        return delta.to(anchor.dtype), gate


class PatchCompensation(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=192, heads=4,
                 layers=tuple(range(1, 13)), mlp_hidden=256,
                 max_gate=0.1, initial_gate=0.01):
        super().__init__()
        if hidden_dim <= 0 or heads <= 0 or hidden_dim % heads:
            raise ValueError('Hidden dimension must be positive and divisible by heads')
        self.layers = tuple(layers)
        if (not self.layers or tuple(sorted(set(self.layers))) != self.layers
                or any(type(i) is not int or not 1 <= i <= 12 for i in self.layers)):
            raise ValueError('Patch layers must be unique increasing integers in [1, 12]')
        self.hidden_dim, self.heads = int(hidden_dim), int(heads)
        self.input_dim, self.head_dim = int(input_dim), hidden_dim // heads
        self.patch_norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in self.layers])
        self.key = nn.Linear(input_dim, hidden_dim, bias=False)
        self.value = nn.Linear(input_dim, hidden_dim, bias=False)
        self.query_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.layer_embedding = nn.Parameter(torch.empty(len(self.layers), hidden_dim))
        nn.init.normal_(self.query_token, std=0.02)
        nn.init.normal_(self.layer_embedding, std=0.02)
        self.grm = GateRefinement(hidden_dim, input_dim, mlp_hidden, max_gate, initial_gate)

    def architecture(self):
        return dict(version=2, input_dim=self.input_dim, hidden_dim=self.hidden_dim,
                    heads=self.heads, layers=list(self.layers), query='patch_only_learned',
                    use_null=False, zero_output=False, mlp_hidden=self.grm.mlp_hidden,
                    max_gate=self.grm.max_gate, initial_gate=self.grm.initial_gate,
                    fusion_scale='detached_anchor_norm')

    def read_local(self, patch_features, return_attention=False):
        if len(patch_features) != len(self.layers):
            raise ValueError('Patch features must follow the configured layer order')
        first = patch_features[0]
        if first.ndim != 3:
            raise ValueError('Patch input must be [batch, tokens, input_dim]')
        batch, tokens, dim = first.shape
        if dim != self.input_dim or tokens < 1 or any(p.shape != first.shape for p in patch_features):
            raise ValueError('All patch layers must have identical valid shapes')
        normalized = [norm(p) for norm,p in zip(self.patch_norms,patch_features)]
        keys = torch.cat([self.key(p) + self.layer_embedding[i][None,None,:]
                          for i,p in enumerate(normalized)],1)
        values = torch.cat([self.value(p) for p in normalized],1)
        query = self.query_token.expand(batch,-1,-1)
        def split(tensor):
            return tensor.reshape(batch,-1,self.heads,self.head_dim).transpose(1,2)
        q,k,v=split(query),split(keys),split(values)
        with torch.autocast(device_type=first.device.type, enabled=False):
            scores=torch.matmul(q.float(),k.float().transpose(-2,-1))*self.head_dim**-0.5
            attention=torch.softmax(scores,-1)
            pooled=torch.matmul(attention,v.float())
        pooled=pooled.transpose(1,2).reshape(batch,self.hidden_dim)
        local=self.grm.refine(pooled.to(first.dtype))
        if not return_attention:
            return local
        attention=attention.squeeze(2).reshape(batch,self.heads,len(self.layers),tokens)
        return local,{'patch_attention':attention,'layer_mass':attention.sum(-1)}

    def forward(self, anchor, patch_features, fusion_enabled=True, return_attention=False):
        result=self.read_local(patch_features,return_attention)
        local,stats=result if return_attention else (result,{})
        delta,gate=self.grm.fuse(anchor,local,fusion_enabled)
        return {'local':local,'delta':delta,'gate':gate,**stats}
