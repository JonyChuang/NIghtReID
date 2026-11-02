"""Shared P12 attention: internal contexts, one final 768-d student identity."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, heads, expansion=4):
        super().__init__()
        if dim % heads:
            raise ValueError('PAD dimension must be divisible by attention heads')
        self.heads = heads
        self.head_dim = dim // heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, expansion * dim), nn.GELU(),
                                 nn.Linear(expansion * dim, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query, memory):
        batch, positions, dim = memory.shape
        q = self.query(query).reshape(batch, 1, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(memory).reshape(batch, positions, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(memory).reshape(batch, positions, self.heads, self.head_dim).transpose(1, 2)
        with torch.autocast(device_type=query.device.type, enabled=False):
            logits = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
            log_attention = F.log_softmax(logits, dim=-1)
            attention = log_attention.exp()
        value = (attention.to(v.dtype) @ v).transpose(1, 2).reshape(batch, dim)
        query = self.norm1(query + self.output(value))
        query = self.norm2(query + self.ffn(query))
        return query, attention[:, :, 0], log_attention[:, :, 0]


class PrototypeAttentionHead(nn.Module):
    """256 is an attention width, never a supervised identity embedding."""
    def __init__(self, input_dim=768, dim=256, heads=8, layers=3, expansion=4):
        super().__init__()
        if input_dim != 768 or layers != 3:
            raise ValueError('This variant uses a 768-d identity space and three PAD layers')
        self.dim = dim
        self.identity_dim = input_dim
        self.spatial_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim))
        self.fusion_query = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim))
        self.student_query = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim))
        self.prototype_query = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim))
        self.layers = nn.ModuleList([CrossAttentionLayer(dim, heads, expansion) for _ in range(layers)])
        self.residual_projection = nn.Linear(dim, input_dim, bias=False)
        nn.init.zeros_(self.residual_projection.weight)

    def prepare(self, patches):
        if patches.ndim != 3 or patches.shape[1:] != (128, 768):
            raise ValueError('PAD requires the 128 raw final-layer P12 tokens')
        return self.spatial_projection(patches)

    def decode(self, query, spatial):
        attention, logs, states = [], [], []
        for layer in self.layers:
            query, a, loga = layer(query, spatial)
            states.append(query)
            attention.append(a)
            logs.append(loga)
        # The post-LN context has norm about sqrt(dim). Keep the residual input
        # at unit scale without introducing an identity projection or another BN.
        return dict(context=query / math.sqrt(self.dim),
                    attention=torch.stack(attention, 1),
                    log_attention=torch.stack(logs, 1), queries=torch.stack(states, 1))

    def fuse(self, global_feature, spatial):
        context = self.decode(self.fusion_query(global_feature), spatial)
        mixed = global_feature + self.residual_projection(context['context'])
        return mixed, context

    def student(self, fused_identity, spatial):
        if fused_identity.ndim != 2 or fused_identity.shape[-1] != self.identity_dim:
            raise ValueError('Student attention must query the final 768-d fusion descriptor')
        result = self.decode(self.student_query(fused_identity), spatial)
        result['unit'] = fused_identity  # The exact retrieval/CNL tensor, not a new embedding.
        return result

    def teacher(self, prototypes, spatial):
        if prototypes.ndim != 2 or prototypes.shape[-1] != self.identity_dim:
            raise ValueError('Teacher prototypes must use the same 768-d identity space')
        return self.decode(self.prototype_query(prototypes.detach()), spatial)


def attention_distillation(student_log_attention, teacher_attention):
    """Spatial KL per image/layer/head; only its teacher target is detached."""
    if student_log_attention.shape != teacher_attention.shape:
        raise ValueError('Teacher/student layer, head and spatial axes must correspond')
    with torch.autocast(device_type=student_log_attention.device.type, enabled=False):
        return F.kl_div(student_log_attention.float(), teacher_attention.detach().float(),
                        reduction='none').sum(-1).mean()
