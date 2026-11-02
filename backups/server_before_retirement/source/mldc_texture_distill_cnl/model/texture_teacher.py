"""An identity-supervised CNN teacher with raw or fixed texture input banks.

This is a task-specific experiment, not an implementation of FSKD or SSTKD.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .backbones.vit_pytorch_mldc import IBN


def cnn_stem():
    return nn.Sequential(nn.Conv2d(3,64,7,2,3,bias=False),IBN(64),nn.ReLU(True),
        nn.Conv2d(64,64,3,1,1,bias=False),IBN(64),nn.ReLU(True),
        nn.Conv2d(64,64,3,1,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(True))


class TextureBank(nn.Module):
    def __init__(self,bank='texture',grid=(16,8)):
        super().__init__()
        if bank not in ('raw','texture'): raise ValueError('Unknown teacher input bank')
        self.bank=bank; self.grid=tuple(grid)
        for name,size,sigma in [('g1',5,1.),('g2',9,2.)]:
            x=torch.arange(size,dtype=torch.float32)-(size-1)/2
            x=torch.exp(-x.square()/(2*sigma*sigma)); x=x/x.sum()
            self.register_buffer(name,(x[:,None]*x[None,:])[None,None].repeat(64,1,1,1))
        dx=torch.tensor([[-1.,0,1],[-2,0,2],[-1,0,1]])/8
        self.register_buffer('dx',dx[None,None].repeat(64,1,1,1))
        self.register_buffer('dy',dx.T[None,None].repeat(64,1,1,1))

    @staticmethod
    def filter(x,kernel):
        p=kernel.shape[-1]//2
        return F.conv2d(F.pad(x,(p,p,p,p),mode='reflect'),kernel,groups=64)

    def forward(self,x):
        if self.bank=='raw': return F.adaptive_avg_pool2d(x,self.grid)
        low1=self.filter(x,self.g1); low2=self.filter(x,self.g2)
        gx=self.filter(low1,self.dx); gy=self.filter(low1,self.dy)
        energy=(x-low1).square()+(low1-low2).square()+gx.square()+gy.square()
        # Pool energies before the square root; avoid cancellation of signed details.
        return F.adaptive_avg_pool2d(energy,self.grid).clamp_min(1e-12).sqrt()


class TextureTeacher(nn.Module):
    def __init__(self,num_classes,bank='texture',grid=(16,8),dim=128):
        super().__init__()
        self.bank_type=bank; self.grid=tuple(grid); self.dim=dim; self.num_classes=num_classes
        self.stem=cnn_stem().requires_grad_(False)
        self.bank=TextureBank(bank,grid)
        self.encoder=nn.Sequential(nn.Conv2d(64,dim,3,padding=1,bias=False),
            nn.GroupNorm(16,dim),nn.ReLU(True),nn.Conv2d(dim,dim,3,padding=1,bias=False),
            nn.GroupNorm(16,dim),nn.ReLU(True))
        self.project=nn.Linear(dim,256,bias=False)
        self.bottleneck=nn.BatchNorm1d(256); self.bottleneck.bias.requires_grad_(False)
        self.classifier=nn.Linear(256,num_classes,bias=False)
        self.provenance={'trained':False,'bank':bank,'seed':1234}

    def train(self,mode=True):
        super().train(mode); self.stem.eval()
        return self

    def initialize_stem(self,path):
        checkpoint=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        state=checkpoint.get('state_dict',checkpoint.get('model',checkpoint))
        selected={}
        for name,value in state.items():
            if name.startswith('module.'): name=name[7:]
            if name.startswith('base.'): name=name[5:]
            if name.startswith('backbone.'): name=name[9:]
            prefix='patch_embed.conv.'
            if name.startswith(prefix): selected[name[len(prefix):]]=value
        self.stem.load_state_dict(selected,strict=True)
        self.provenance['stem_source']=str(path)

    def forward(self,images):
        with torch.no_grad():
            spatial=self.bank(self.stem(images))
        spatial=self.encoder(spatial)
        raw=self.project(spatial.mean((2,3)))
        neck=self.bottleneck(raw)
        return dict(spatial=spatial,raw=raw,unit=F.normalize(neck.float(),dim=1),
            logits=self.classifier(neck))

    def load_trained(self,path,expected_epochs=40):
        checkpoint=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        meta=checkpoint['teacher_metadata']
        if (meta.get('stage')!='identity_texture_teacher_v1' or not meta.get('trained')
                or meta.get('bank')!=self.bank_type or meta.get('seed')!=1234
                or meta.get('epoch')!=expected_epochs or meta.get('num_classes')!=self.num_classes
                or meta.get('optimizer_updates',0)<=0):
            raise ValueError('Use the matching, completed, identity-supervised teacher checkpoint')
        self.load_state_dict(checkpoint['state_dict'],strict=True)
        self.provenance=dict(meta,checkpoint=str(Path(path)))
        return self


def distillation_terms(student,teacher):
    """Align normalized features, within-image patch relations and spatial energy."""
    s=student.float().flatten(2).transpose(1,2)
    t=teacher.detach().float().flatten(2).transpose(1,2)
    if s.shape!=t.shape: raise ValueError('Student/teacher aligned grids differ')
    sn=F.normalize(s,dim=-1); tn=F.normalize(t,dim=-1)
    feature=(sn-tn).square().mean()
    rs=sn@sn.transpose(1,2); rt=tn@tn.transpose(1,2)
    mask=~torch.eye(s.shape[1],dtype=torch.bool,device=s.device)
    relation=(rs-rt).square()[:,mask].mean()
    sa=s.square().mean(-1)+1e-8; ta=t.square().mean(-1)+1e-8
    sa=sa/sa.sum(1,keepdim=True); ta=ta/ta.sum(1,keepdim=True)
    attention=(ta*(ta.log()-sa.log())).sum(1).mean().clamp_min(0)
    return dict(feature=feature,relation=relation,attention=attention)
