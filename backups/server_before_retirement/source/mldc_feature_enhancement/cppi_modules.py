"""CoMer-inspired CNN/MRFP and fixed-neighborhood CTI-Lite, no custom CUDA ops.

Independent adaptation of https://github.com/Traffic-X/ViT-CoMer. This is not
the paper's bidirectional deformable CTI or its dense prediction backbone.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def group_norm(channels):
    return nn.GroupNorm(math.gcd(32, channels), channels)


class ConvStage(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1, bias=False),
                         group_norm(outgoing), nn.GELU())


class MultiReceptivePyramid(nn.Module):
    """Shared reduce/expand projections; per-scale channel-split 3/5 DWConv."""
    def __init__(self, dim=192):
        super().__init__()
        if dim % 4:
            raise ValueError('MRFP width must be divisible by four')
        hidden = dim // 2
        self.reduce = nn.Conv2d(dim, hidden, 1, bias=False)
        self.dw3 = nn.ModuleList(nn.Conv2d(hidden//2, hidden//2, 3, padding=1,
                                         groups=hidden//2) for _ in range(3))
        self.dw5 = nn.ModuleList(nn.Conv2d(hidden//2, hidden//2, 5, padding=2,
                                         groups=hidden//2) for _ in range(3))
        self.norms = nn.ModuleList(group_norm(hidden) for _ in range(3))
        self.expand = nn.Conv2d(hidden, dim, 1, bias=False)

    def forward(self, pyramid):
        result = []
        for i, source in enumerate(pyramid):
            a, b = self.reduce(source).chunk(2, dim=1)
            joined = torch.cat((self.dw3[i](a), self.dw5[i](b)), 1)
            result.append(self.expand(F.gelu(self.norms[i](joined))))
        return tuple(result)


class CNNPyramid(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.stem = nn.Sequential(ConvStage(3,48), ConvStage(48,96), ConvStage(96,dim))
        self.stage4 = ConvStage(dim,dim)
        self.stage5 = ConvStage(dim,dim)
        self.mrfp = MultiReceptivePyramid(dim)

    def forward(self, images):
        c3 = self.stem(images); c4 = self.stage4(c3); c5 = self.stage5(c4)
        return self.mrfp((c3,c4,c5))


class PatchPyramid(nn.Module):
    """Source-control: same CTI reads a pyramid derived from current ViT patches."""
    def __init__(self, input_dim=768, dim=192):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.project = nn.Linear(input_dim,dim,bias=False)
        self.mrfp = MultiReceptivePyramid(dim)

    def forward(self, patches, grid):
        x = self.project(self.norm(patches)).transpose(1,2).reshape(len(patches),-1,*grid)
        h,w = grid
        return self.mrfp((F.interpolate(x,size=(h*2,w*2),mode='bilinear',align_corners=False),
                          x,F.interpolate(x,size=(max(1,h//2),max(1,w//2)),mode='bilinear',align_corners=False)))


class CNNTokenReader(nn.Module):
    """A single SDCL-style readout block over unpartitioned CNN pyramid tokens."""
    def __init__(self,cnn_dim,output_dim,block,norm):
        super().__init__()
        if output_dim%4:
            raise ValueError('2D position encoding requires a dimension divisible by four')
        self.project=nn.Sequential(nn.LayerNorm(cnn_dim),nn.Linear(cnn_dim,output_dim))
        self.scale_embed=nn.Parameter(torch.empty(3,1,output_dim))
        nn.init.normal_(self.scale_embed,std=.02)
        self.block=block;self.norm=norm;self.output_dim=output_dim

    def position(self,height,width,device):
        # Coordinates describe image positions at every scale; no anatomical prior.
        yy,xx=torch.meshgrid((torch.arange(height,device=device,dtype=torch.float32)+.5)/height,
                             (torch.arange(width,device=device,dtype=torch.float32)+.5)/width,indexing='ij')
        freq=10000.**(-torch.arange(self.output_dim//4,device=device,dtype=torch.float32)/(self.output_dim//4))
        x=xx.reshape(-1,1)*2*math.pi*freq
        y=yy.reshape(-1,1)*2*math.pi*freq
        return torch.cat((x.sin(),x.cos(),y.sin(),y.cos()),dim=1)

    def make_tokens(self,pyramid):
        if len(pyramid)!=3:
            raise ValueError('Expected three CNN scales')
        result=[]
        for level,feature in enumerate(pyramid):
            tokens=self.project(feature.flatten(2).transpose(1,2))
            position=self.position(*feature.shape[-2:],feature.device)
            result.append(tokens+position.to(tokens.dtype)+self.scale_embed[level].to(tokens.dtype))
        return torch.cat(result,dim=1)

    def read(self,query,tokens):
        sequence=torch.cat((query[:,None].to(tokens.dtype),tokens),dim=1)
        return self.norm(self.block(sequence))[:,0]


class CNNCrossAttentionReader(CNNTokenReader):
    """One CLS query attends only to CNN keys/values, using SSL block weights."""
    def read(self,query,tokens):
        x=query[:,None].to(tokens.dtype)
        block=self.block; attn=block.attn
        dim=self.output_dim; heads=attn.num_heads
        weight=attn.qkv.weight; bias=attn.qkv.bias
        q=F.linear(block.norm1(x),weight[:dim],None if bias is None else bias[:dim])
        kv=F.linear(block.norm1(tokens),weight[dim:],None if bias is None else bias[dim:])
        q=q.reshape(len(x),1,heads,dim//heads).transpose(1,2)
        k,v=kv.reshape(len(x),tokens.shape[1],2,heads,dim//heads).permute(2,0,3,1,4).unbind(0)
        attention=attn.attn_drop(((q @ k.transpose(-2,-1))*attn.scale).softmax(dim=-1))
        context=(attention @ v).transpose(1,2).reshape(len(x),1,dim)
        x=x+block.drop_path(attn.proj_drop(attn.proj(context)))
        x=x+block.drop_path(block.mlp(block.norm2(x)))
        return self.norm(x)[:,0]


def local_sample(maps, grid):
    """Nine positions per scale; image-normalized patch centers, no body regions.

    Border interpolation is defined inside the image; candidates whose centers
    lie outside the image are masked out, not treated as zero-valued neighbors.
    """
    b = maps[0].shape[0]; device = maps[0].device
    h,w = grid
    yy,xx = torch.meshgrid((torch.arange(h,device=device,dtype=torch.float32)+.5)/h,
                          (torch.arange(w,device=device,dtype=torch.float32)+.5)/w,indexing='ij')
    centers = torch.stack((xx,yy),-1).reshape(1,h*w,1,2)
    dy,dx = torch.meshgrid(torch.arange(-1,2,device=device),torch.arange(-1,2,device=device),indexing='ij')
    offsets = torch.stack((dx,dy),-1).reshape(1,1,9,2).float()
    sampled,valid = [],[]
    # FP32 grid_sample supports CPU verification and avoids half-precision
    # coordinate quantization. Casting inputs keeps autograd connected.
    with torch.autocast(device_type=device.type,enabled=False):
        for source in maps:
            hs,ws = source.shape[-2:]
            points = centers + offsets / centers.new_tensor([ws,hs])
            mask = ((points>=0)&(points<=1)).all(-1)
            values = F.grid_sample(source.float(),(2*points-1).expand(b,-1,-1,-1),
                mode='bilinear',padding_mode='border',align_corners=False)
            sampled.append(values.permute(0,2,3,1));valid.append(mask)
    return torch.cat(sampled,2),torch.cat(valid,2)


class CTILite(nn.Module):
    def __init__(self,input_dim=768,dim=192,heads=4,initial_ratio=.05,up_init_std=.01):
        super().__init__()
        if heads < 1 or dim % heads or not math.isfinite(initial_ratio) or initial_ratio <= 0 or up_init_std <= 0:
            raise ValueError('Invalid CTI head count, width, magnitude, or initialization')
        self.dim,self.heads,self.initial_ratio = dim,heads,float(initial_ratio)
        # One positive learned scalar per stage, with no upper clamp. The
        # optimizer excludes it from weight decay, so task loss controls it.
        initial_logit = initial_ratio + math.log(-math.expm1(-initial_ratio))
        self.budget_logit = nn.Parameter(torch.tensor(initial_logit,dtype=torch.float32))
        self.query_norm = nn.LayerNorm(input_dim)
        self.query = nn.Linear(input_dim,dim,bias=False)
        self.source_norm = nn.LayerNorm(dim)
        self.key = nn.Linear(dim,dim,bias=False)
        self.value = nn.Linear(dim,dim,bias=False)
        self.level_embed = nn.Parameter(torch.empty(3,dim))
        self.relative_bias = nn.Parameter(torch.zeros(heads,27))
        self.up = nn.Linear(dim,input_dim,bias=False)
        nn.init.normal_(self.level_embed,std=.02)
        nn.init.normal_(self.up.weight,std=up_init_std)

    def correction_budget(self):
        return F.softplus(self.budget_logit.float())

    def forward(self,patches,pyramid,grid,diagnostics=False):
        b,n,_ = patches.shape
        if n != grid[0]*grid[1] or len(pyramid)!=3:
            raise ValueError('CTI expects a matching patch grid and three feature scales')
        sources = []
        for i,feature in enumerate(pyramid):
            tokens = feature.flatten(2).transpose(1,2)
            z = self.source_norm(tokens + self.level_embed[i])
            kv = torch.cat((self.key(z),self.value(z)),-1)
            sources.append(kv.transpose(1,2).reshape(b,self.dim*2,*feature.shape[-2:]))
        neighbors,valid = local_sample(sources,grid)
        k,v = neighbors.chunk(2,dim=-1)
        width = self.dim//self.heads
        q = self.query(self.query_norm(patches)).reshape(b,n,self.heads,width)
        k = k.reshape(b,n,27,self.heads,width).permute(0,1,3,2,4)
        v = v.reshape(b,n,27,self.heads,width).permute(0,1,3,2,4)
        with torch.autocast(device_type=patches.device.type,enabled=False):
            logits = torch.einsum('bnhd,bnhkd->bnhk',q.float(),k.float()) / math.sqrt(width)
            logits = logits + self.relative_bias[None,None].float()
            attention = logits.masked_fill(~valid[:,:,None,:],float('-inf')).softmax(-1)
            readout = torch.einsum('bnhk,bnhkd->bnhd',attention,v.float()).reshape(b,n,self.dim)
        raw = self.up(readout)
        with torch.autocast(device_type=patches.device.type,enabled=False):
            original,direction = patches.float(),raw.float()
            source_rms = original.square().mean((1,2),keepdim=True).sqrt()
            delta_rms = direction.square().mean((1,2),keepdim=True).clamp_min(1e-16).sqrt()
            budget = self.correction_budget()
            correction = budget*source_rms*direction/delta_rms
            changed = original+correction
        stats = {}
        if diagnostics:
            a,c,d = original.detach(),changed.detach(),correction.detach()
            weights = attention.detach()
            entropy = -(weights*weights.clamp_min(1e-30).log()).sum(-1).mean(-1)
            entropy = entropy/valid.sum(-1).float().clamp_min(2).log()
            stats = dict(before_norm=a.norm(dim=-1).reshape(b,*grid),after_norm=c.norm(dim=-1).reshape(b,*grid),
                correction_norm=d.norm(dim=-1).reshape(b,*grid),
                correction_ratio=d.flatten(1).norm(dim=-1)/a.flatten(1).norm(dim=-1).clamp_min(1e-12),
                budget=budget.detach().expand(b),
                scale_mass=weights.reshape(b,n,self.heads,3,9).sum(-1).mean((1,2)),
                local_entropy=entropy.reshape(b,*grid))
        return changed,stats
