"""Identity-supervised cross-layer patch readout with a zero-valued null option."""
import math
import torch
from torch import nn


class PatchCompensation(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=192, heads=4,
                 layers=tuple(range(1, 13)), use_null=True, zero_output=True):
        super().__init__()
        if hidden_dim <= 0 or heads <= 0 or hidden_dim % heads:
            raise ValueError('Patch hidden dimension must be positive and divisible by heads')
        self.layers = tuple(layers)
        if (not self.layers or tuple(sorted(set(self.layers))) != self.layers
                or any(type(i) is not int or not 1 <= i <= 12 for i in self.layers)):
            raise ValueError('Patch layers must be unique increasing integers in [1, 12]')
        self.hidden_dim, self.heads = hidden_dim, heads
        self.head_dim = hidden_dim // heads
        self.use_null, self.zero_output = bool(use_null), bool(zero_output)
        self.patch_norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in self.layers])
        self.global_norm = nn.LayerNorm(input_dim)
        self.query = nn.Linear(input_dim, hidden_dim, bias=False)
        self.key = nn.Linear(input_dim, hidden_dim, bias=False)
        self.value = nn.Linear(input_dim, hidden_dim, bias=False)
        self.query_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.layer_embedding = nn.Parameter(torch.empty(len(self.layers), hidden_dim))
        if self.use_null:
            self.null_key = nn.Parameter(torch.empty(1, 1, hidden_dim))
            self.null_bias = nn.Parameter(torch.zeros(heads))
        else:
            self.register_parameter('null_key', None)
            self.register_parameter('null_bias', None)
        # No output bias: selecting a zero value must produce zero compensation.
        self.output = nn.Linear(hidden_dim, input_dim, bias=False)
        nn.init.normal_(self.query_token, std=0.02)
        nn.init.normal_(self.layer_embedding, std=0.02)
        if self.null_key is not None:
            nn.init.normal_(self.null_key, std=0.02)
        if self.zero_output:
            nn.init.zeros_(self.output.weight)

    def architecture(self):
        return dict(layers=list(self.layers), hidden_dim=self.hidden_dim,
                    heads=self.heads, use_null=self.use_null,
                    zero_output=self.zero_output, input_dim=self.output.out_features,
                    null_token_count_prior=True, output_bias=False)

    def forward(self, global_feature, patch_features, return_attention=False):
        if len(patch_features) != len(self.layers):
            raise ValueError('Patch features must follow the configured layer order')
        batch, dim = global_feature.shape
        if any(p.ndim != 3 or p.shape[0] != batch or p.shape[2] != dim
               or p.shape[1] != patch_features[0].shape[1] for p in patch_features):
            raise ValueError('Expected equal-sized [batch, patch, input_dim] tensors')
        tokens_per_layer = patch_features[0].shape[1]
        normalized = [norm(p) for norm, p in zip(self.patch_norms, patch_features)]
        keys = torch.cat([self.key(p) + self.layer_embedding[i][None, None, :]
                          for i, p in enumerate(normalized)], dim=1)
        values = torch.cat([self.value(p) for p in normalized], dim=1)
        query = self.query(self.global_norm(global_feature))[:, None, :] + self.query_token
        if self.use_null:
            # The null value is fixed at zero and is never a trainable parameter.
            keys = torch.cat([self.null_key.expand(batch, -1, -1).to(keys.dtype), keys], dim=1)
            values = torch.cat([values.new_zeros(batch, 1, self.hidden_dim), values], dim=1)
        def split(tensor):
            return tensor.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = split(query), split(keys), split(values)
        # FP32 softmax protects low-light attention scores under autocast.
        with torch.autocast(device_type=global_feature.device.type, enabled=False):
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.head_dim ** -0.5
            if self.use_null:
                # One null option competes against N real tokens. A log(N) prior
                # gives equal null/real total mass when all content logits match.
                # Otherwise initial null mass would be only 1/(N+1).
                null_score = scores[..., :1] + math.log(keys.shape[1] - 1)
                null_score = null_score + self.null_bias[None, :, None, None]
                scores = torch.cat([null_score, scores[..., 1:]], dim=-1)
            attention = torch.softmax(scores, dim=-1)
            local = torch.matmul(attention, v.float())
        local = local.transpose(1, 2).reshape(batch, self.hidden_dim)
        delta = self.output(local.to(global_feature.dtype))
        if not return_attention:
            return delta
        attention = attention.squeeze(2)
        offset = int(self.use_null)
        null_mass = attention[:, :, 0] if self.use_null else attention.new_zeros(batch, self.heads)
        patch_attention = attention[:, :, offset:].reshape(
            batch, self.heads, len(self.layers), tokens_per_layer)
        diagnostics = {'null_mass': null_mass,
                       'layer_mass': patch_attention.sum(-1),
                       'patch_attention': patch_attention}
        return delta, diagnostics
