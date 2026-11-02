"""Image-conditioned shared PAD head. P12 is the only spatial K/V source."""
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
        # Distillation uses normalized, pre-dropout probabilities in FP32.
        with torch.autocast(device_type=query.device.type, enabled=False):
            logits = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
            log_attention = F.log_softmax(logits, dim=-1)
            attention = log_attention.exp()
        value = (attention.to(v.dtype) @ v).transpose(1, 2).reshape(batch, dim)
        query = self.norm1(query + self.output(value))
        query = self.norm2(query + self.ffn(query))
        return query, attention[:, :, 0], log_attention[:, :, 0]


class PrototypeAttentionHead(nn.Module):
    def __init__(self, input_dim=768, dim=256, heads=8, layers=3, expansion=4):
        super().__init__()
        if layers != 3:
            raise ValueError('This experiment fixes three sequential PAD layers')
        self.dim = dim
        self.spatial_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim))
        self.student_query = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim))
        self.prototype_query = nn.Linear(dim, dim)
        self.layers = nn.ModuleList([CrossAttentionLayer(dim, heads, expansion) for _ in range(layers)])
        # PAD Eq. (8): the same FC+BN projection is used by both query paths.
        self.embedding = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim))
        self.residual_projection = nn.Linear(dim, input_dim, bias=False)
        nn.init.zeros_(self.residual_projection.weight)

    def prepare(self, patches):
        if patches.ndim != 3 or patches.shape[1] != 128 or patches.shape[2] != 768:
            raise ValueError('PAD requires the 128 raw final-layer P12 tokens')
        return self.spatial_projection(patches)

    def decode(self, query, spatial):
        attention, logs, states = [], [], []
        for layer in self.layers:
            query, a, loga = layer(query, spatial)
            states.append(query)
            attention.append(a)
            logs.append(loga)
        with torch.autocast(device_type=query.device.type, enabled=False):
            embedding = self.embedding(query.float()).float()
            unit = F.normalize(embedding, dim=-1)
        return dict(unit=unit, embedding=embedding, attention=torch.stack(attention, 1),
                    log_attention=torch.stack(logs, 1), queries=torch.stack(states, 1))

    def student(self, global_feature, spatial):
        return self.decode(self.student_query(global_feature), spatial)

    def teacher(self, prototypes, spatial):
        return self.decode(self.prototype_query(prototypes.detach()), spatial)


def attention_distillation(student_log_attention, teacher_attention):
    """Equation (12): sum spatial KL, then mean over images, layers and heads."""
    if student_log_attention.shape != teacher_attention.shape:
        raise ValueError('Teacher/student layer, head and spatial axes must correspond')
    with torch.autocast(device_type=student_log_attention.device.type, enabled=False):
        return F.kl_div(student_log_attention.float(), teacher_attention.detach().float(),
                        reduction='none').sum(-1).mean()
