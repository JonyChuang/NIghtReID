"""The existing DCM ConvStream, specialized as a training-only texture teacher."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .convstream import ConvStream


def exposure_view(images,gain=.7,gamma=1.):
    """Photometric changes only; input/output use the original mean=.5,std=.5."""
    rgb=(images*.5+.5).clamp(0,1)
    return (rgb.clamp_min(1e-6).pow(gamma)*gain).clamp(0,1)*2-1


class EnergyBank(nn.Module):
    def __init__(self,channels,bank,grid):
        super().__init__()
        self.channels=channels; self.bank=bank; self.grid=tuple(grid)
        if bank not in ('raw','texture'): raise ValueError('Unknown ConvStream input bank')
        for name,size,sigma in [('g1',5,1.),('g2',9,2.)]:
            axis=torch.arange(size,dtype=torch.float32)-(size-1)/2
            one=torch.exp(-axis.square()/(2*sigma*sigma)); one/=one.sum()
            self.register_buffer(name,(one[:,None]*one[None,:])[None,None].repeat(channels,1,1,1))
        dx=torch.tensor([[-1.,0,1],[-2,0,2],[-1,0,1]])/8
        self.register_buffer('dx',dx[None,None].repeat(channels,1,1,1))
        self.register_buffer('dy',dx.T[None,None].repeat(channels,1,1,1))

    def filter(self,x,kernel):
        p=kernel.shape[-1]//2
        return F.conv2d(F.pad(x,(p,p,p,p),mode='reflect'),kernel,groups=self.channels)

    def forward(self,x):
        # Normalize each channel's scale equally in raw and texture controls.
        scale=x.square().mean((2,3),keepdim=True).add(1e-6).sqrt()
        if self.bank=='raw': return F.adaptive_avg_pool2d(x/scale,self.grid)
        low1=self.filter(x,self.g1); low2=self.filter(x,self.g2)
        gx=self.filter(low1,self.dx); gy=self.filter(low1,self.dy)
        energy=(x-low1).square()+(low1-low2).square()+gx.square()+gy.square()
        return F.adaptive_avg_pool2d(energy,self.grid).clamp_min(1e-12).sqrt()/scale


class TextureTeacher(nn.Module):
    def __init__(self,num_classes,bank='texture',grid=(8,4),dim=128):
        super().__init__()
        self.bank_type=bank; self.grid=tuple(grid); self.dim=dim; self.num_classes=num_classes
        self.convstream=ConvStream(in_chans=3)
        self.banks=nn.ModuleList([EnergyBank(c,bank,grid) for c in [48,96,192]])
        self.encoder=nn.Sequential(nn.Conv2d(336,dim,1,bias=False),nn.GroupNorm(16,dim))
        self.project=nn.Linear(dim,256,bias=False)
        self.bottleneck=nn.BatchNorm1d(256); self.bottleneck.bias.requires_grad_(False)
        self.classifier=nn.Linear(256,num_classes,bias=False)
        self.provenance={'trained':False,'bank':bank,'seed':1234}

    def train(self,mode=True):
        super().train(mode)
        # Fine-tune convolution weights while retaining the old DCM BN statistics.
        for module in self.convstream.modules():
            if isinstance(module,nn.BatchNorm2d): module.eval()
        return self

    def initialize_convstream(self,path):
        import yaml
        checkpoint=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        state=checkpoint.get('state_dict',checkpoint.get('model',checkpoint))
        state={k[7:] if k.startswith('module.') else k:v for k,v in state.items()}
        prefix='detail_branch.convstream.'
        selected={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
        if not selected and any(k.startswith('convs.') for k in state):
            selected={k:v for k,v in state.items() if k.startswith('convs.')}
        config=checkpoint.get('config')
        if isinstance(config,str):
            config=yaml.safe_load(config)
            if config['DATASETS']['NAMES']!='nightreid' or config['SOLVER']['SEED']!=1234:
                raise ValueError('Use the existing NightReID seed1234 DCM ConvStream')
        if checkpoint.get('epoch')!=40:
            raise ValueError('Expected the previously used DCM epoch40 starting checkpoint')
        self.convstream.load_state_dict(selected,strict=True)
        self.provenance.update(warmstart=str(path),warmstart_epoch=40,
            warmstart_scope='DCM ConvStream 48/96/192 only; no ViT or fusion parameters')

    def spatial(self,images):
        maps=self.convstream(images)
        banks=[bank(maps['D'+str(i+1)]) for i,bank in enumerate(self.banks)]
        bank=torch.cat(banks,dim=1)
        return self.encoder(bank),bank

    def forward(self,images):
        spatial,bank=self.spatial(images)
        raw=self.project(spatial.mean((2,3))); neck=self.bottleneck(raw)
        return dict(spatial=spatial,tokens=F.normalize(spatial.flatten(2).transpose(1,2),dim=-1),
            raw=raw,unit=F.normalize(neck.float(),dim=1),logits=self.classifier(neck))

    @torch.no_grad()
    def describe(self,images,exposure=.7):
        if self.training: raise RuntimeError('Only the frozen eval teacher supplies student targets')
        spatial,bank=self.spatial(images)
        other,_=self.spatial(exposure_view(images,exposure))
        tokens=F.normalize(spatial.flatten(2).transpose(1,2),dim=-1)
        changed=F.normalize(other.flatten(2).transpose(1,2),dim=-1)
        stability=(tokens*changed).sum(-1).clamp(0,1)
        signal=bank.square().mean(1).clamp_min(1e-12).sqrt().flatten(1)
        confidence=stability*signal/(signal+.05)
        return dict(tokens=tokens,confidence=confidence)

    def load_trained(self,path,expected_epochs=10):
        checkpoint=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        meta=checkpoint.get('teacher_metadata',{})
        expected=dict(stage='convstream_identity_texture_teacher_v2',trained=True,bank=self.bank_type,
            seed=1234,epoch=expected_epochs,num_classes=self.num_classes,grid=list(self.grid),dim=self.dim)
        if any(meta.get(k)!=v for k,v in expected.items()) or meta.get('optimizer_updates',0)<=0:
            raise ValueError('Use the completed, matched ConvStream texture teacher')
        self.load_state_dict(checkpoint['state_dict'],strict=True)
        self.provenance=dict(meta,checkpoint=str(Path(path)))
        return self


def specialization_consistency(first,second):
    a=first['tokens']; b=second['tokens']
    illumination=(1-(a*b).sum(-1)).mean()
    mask=~torch.eye(a.shape[1],device=a.device,dtype=torch.bool)
    structure=((a@a.transpose(1,2)-b@b.transpose(1,2)).square())[:,mask].mean()
    return illumination,structure
