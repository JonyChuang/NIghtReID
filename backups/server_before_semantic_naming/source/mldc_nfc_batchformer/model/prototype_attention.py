"""Direct per-level CLS/prototype cross-attention; no decoder stack or FFN."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LayerPrototypeRefiner(nn.Module):
    """One direct cross-attention, shared by student and teacher at this level."""
    def __init__(self,dim=256,heads=8):
        super().__init__()
        if dim%heads:
            raise ValueError('Attention width must be divisible by head count')
        self.dim=dim;self.heads=heads;self.head_dim=dim//heads
        self.query_norm=nn.LayerNorm(768)
        self.patch_norm=nn.LayerNorm(768)
        self.query_projection=nn.Linear(768,dim)
        self.key_projection=nn.Linear(768,dim)
        self.value_projection=nn.Linear(768,dim)
        self.output_projection=nn.Linear(dim,768,bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def prepare(self,patches):
        if patches.ndim!=3 or patches.shape[1:]!=(128,768):
            raise ValueError('Each level requires 128 raw 768-d post-block patches')
        memory=self.patch_norm(patches)
        b,n,_=memory.shape
        key=self.key_projection(memory).reshape(b,n,self.heads,self.head_dim).transpose(1,2)
        value=self.value_projection(memory).reshape(b,n,self.heads,self.head_dim).transpose(1,2)
        return dict(key=key,value=value)

    def forward(self,query,raw_cls,spatial):
        b=len(query)
        q=self.query_projection(self.query_norm(query)).reshape(b,self.heads,1,self.head_dim)
        with torch.autocast(device_type=query.device.type,enabled=False):
            logits=(q.float()@spatial['key'].float().transpose(-1,-2))/math.sqrt(self.head_dim)
            log_attention=F.log_softmax(logits,dim=-1)
            attention=log_attention.exp()
        value=spatial['value']
        context=(attention.to(value.dtype)@value).transpose(1,2).reshape(b,self.dim)
        # The prototype only supplies Q. There is no prototype-query residual
        # or FFN in the visual context; V comes exclusively from this image.
        refined=raw_cls+self.output_projection(context)
        return dict(refined_cls=refined,context=context,
                    attention=attention[:,:,0],log_attention=log_attention[:,:,0])


class PrototypeAttentionHead(nn.Module):
    """Four independent attention heads; one shared identity prototype input."""
    def __init__(self,input_dim=768,dim=256,heads=8,layers=1,
                 feature_layers=(9,10,11,12)):
        super().__init__()
        self.feature_layers=tuple(feature_layers)
        if input_dim!=768 or self.feature_layers!=(9,10,11,12) or layers!=1:
            raise ValueError('Use one direct cross-attention per backbone level 9-12')
        self.level_heads=nn.ModuleDict({str(level):LayerPrototypeRefiner(dim,heads)
                                       for level in self.feature_layers})

    def _collect(self,raw_layers,states):
        if raw_layers.ndim!=3 or raw_layers.shape[0]!=12 or raw_layers.shape[-1]!=768:
            raise ValueError('Expected all twelve raw CLS layers')
        refined=torch.stack([states[i+1]['refined_cls'] if i+1 in states else raw_layers[i]
                             for i in range(12)],dim=0)
        return dict(refined_layers=refined,levels=states,
                    attention=torch.stack([states[l]['attention'] for l in self.feature_layers],1),
                    log_attention=torch.stack([states[l]['log_attention'] for l in self.feature_layers],1))

    def student(self,raw_layers,patches):
        if tuple(sorted(patches))!=self.feature_layers:
            raise ValueError('Need separately captured P9, P10, P11 and P12')
        states={};spatial={}
        for level in self.feature_layers:
            head=self.level_heads[str(level)]
            spatial[level]=head.prepare(patches[level])
            states[level]=head(raw_layers[level-1],raw_layers[level-1],spatial[level])
        result=self._collect(raw_layers,states);result['spatial']=spatial
        return result

    def teacher(self,prototype,raw_layers,spatial):
        if prototype.ndim!=2 or prototype.shape[-1]!=768:
            raise ValueError('All four levels must read the same 768-d identity prototype')
        prototype=prototype.detach()
        states={}
        for level in self.feature_layers:
            states[level]=self.level_heads[str(level)](
                prototype,raw_layers[level-1],spatial[level])
        return self._collect(raw_layers,states)


def attention_distillation(student_log_attention,teacher_attention):
    """Mean spatial KL over batch, four backbone levels and attention heads."""
    if student_log_attention.shape!=teacher_attention.shape:
        raise ValueError('Teacher and student must align in level, head and patch axes')
    with torch.autocast(device_type=student_log_attention.device.type,enabled=False):
        return F.kl_div(student_log_attention.float(),teacher_attention.detach().float(),
                        reduction='none').sum(-1).mean()
