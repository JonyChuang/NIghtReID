"""Interface and gradient verification with a small substitute backbone only."""
import json
import torch
from torch import nn
from .model import JointLocalReID, fuse_local
from .loss import objectives


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__(); self.linear=nn.Linear(32,32)
    def forward(self,x):
        return x + 0.1*self.linear(x+x.mean(1,keepdim=True))


class TinyBase(nn.Module):
    in_planes=32
    def __init__(self):
        super().__init__();self.stem=nn.Conv2d(3,32,4,4);self.blocks=nn.ModuleList(TinyBlock() for _ in range(12))
    def forward(self,x,**kwargs):
        x=self.stem(x).flatten(2).transpose(1,2)
        x=torch.cat((x.mean(1,keepdim=True),x),1);features=[]
        for block in self.blocks:
            x=block(x);features.append(x[:,0])
        return torch.stack(features)


class TinyMLDC(nn.Module):
    def __init__(self):
        super().__init__();self.layer_weights=nn.Parameter(torch.zeros(12))
    def forward(self,x):
        return (x[8:]*self.layer_weights[8:].softmax(0)[:,None,None]).sum(0)


class TinyGlobal(nn.Module):
    def __init__(self):
        super().__init__();self.base=TinyBase();self.mldc=TinyMLDC();self.bottleneck=nn.BatchNorm1d(32)
        self.classifier=nn.Linear(32,4,bias=False);self.dropout=nn.Identity();self.neck_feat='after'


def main():
    torch.manual_seed(1234);torch.set_num_threads(1)
    report={}
    for mode in ('cosine','mean'):
        model=JointLocalReID(TinyGlobal(),4,dim=16,heads=4,mode=mode)
        model.train();x=torch.randn(8,3,16,16);targets=torch.arange(4).repeat_interleave(2)
        before=model.global_model.bottleneck.running_mean.clone()
        data=model(x);loss,terms=objectives(data,targets)
        loss.backward()
        for name in ('global_model.base.stem.weight','global_model.base.blocks.3.linear.weight',
                     'global_model.base.blocks.5.linear.weight','global_model.base.blocks.7.linear.weight',
                     'global_model.mldc.layer_weights','global_model.bottleneck.weight',
                     'global_model.classifier.weight','reader.values.0.weight','local_classifier.weight'):
            p=dict(model.named_parameters())[name]
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0,name
        if mode=='cosine':
            for p in (model.reader.query,model.reader.keys[0].weight):
                assert p.grad is not None and p.grad.norm()>0
        assert not torch.equal(before,model.global_model.bottleneck.running_mean)
        assert all(block.training for block in model.global_model.base.blocks)
        assert data['attention'].shape==(8,4,3,16)
        assert torch.allclose(data['attention'].sum(-1),torch.ones(8,4,3),atol=1e-6)
        ratio=data['delta'].norm(dim=-1)/data['global_feature'].norm(dim=-1)
        assert torch.allclose(ratio,torch.full_like(ratio,0.1),atol=1e-6)
        assert all(len(block._forward_hooks)==0 for block in model.global_model.base.blocks)
        saved={k:v.clone() for k,v in model.state_dict().items()}
        replica=JointLocalReID(TinyGlobal(),4,dim=16,heads=4,mode=mode);replica.load_state_dict(saved,strict=True)
        model.eval();replica.eval()
        with torch.no_grad():assert torch.allclose(model(x),replica(x),atol=1e-6)
        model.train();model.zero_grad(set_to_none=True)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            amp_data=model(x);amp_loss,_=objectives(amp_data,targets)
        amp_loss.backward()
        assert torch.isfinite(amp_loss) and model.reader.values[0].weight.grad.norm()>0
        report[mode]={'interface':True,'joint_gradients':True,'bn_updates':True,'checkpoint_roundtrip':True,'cpu_bfloat16_autocast':True}
    # Check the full gradient through ||g||, independently of backbone details.
    g=torch.randn(3,32,requires_grad=True);a=torch.randn(3,32,requires_grad=True)
    _,delta=fuse_local(g,a);norm_grad=torch.autograd.grad(delta.square().sum(),g)[0]
    assert torch.allclose(norm_grad,0.02*g,rtol=1e-5,atol=1e-6)
    report['differentiable_global_norm']=True
    report['scope']='Small substitute backbone, no real ViT, dataset, optimizer update, or training result'
    print(json.dumps(report))


if __name__=='__main__':main()
