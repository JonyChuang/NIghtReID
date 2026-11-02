"""Verify BoQ joint gradients with a small substitute backbone."""
import json
from contextlib import nullcontext
import torch
from torch import nn
from .boq import BoQBlock
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


def verify_device(device, amp_dtype=None):
    model=JointLocalReID(TinyGlobal(),4,dim=16,heads=4,queries=4).to(device)
    model.train();x=torch.randn(8,3,16,16,device=device);targets=torch.arange(4,device=device).repeat_interleave(2)
    before=model.global_model.bottleneck.running_mean.clone()
    context=torch.autocast(device.type,dtype=amp_dtype) if amp_dtype else nullcontext()
    with context:
        data=model(x);loss,_=objectives(data,targets)
    loss.backward()
    selected=('global_model.base.stem.weight','global_model.base.blocks.3.linear.weight',
              'global_model.base.blocks.5.linear.weight','global_model.base.blocks.7.linear.weight',
              'global_model.mldc.layer_weights','global_model.bottleneck.weight',
              'global_model.classifier.weight','reader.projections.0.weight',
              'reader.blocks.0.queries','reader.blocks.0.query_attention.in_proj_weight',
              'reader.blocks.0.cross_attention.in_proj_weight',
              'reader.blocks.0.encoder.linear1.weight','reader.slot_projection.weight','local_classifier.weight')
    for name in selected:
        p=dict(model.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0,name
    assert torch.isfinite(loss)
    assert not torch.equal(before,model.global_model.bottleneck.running_mean)
    assert all(block.training for block in model.global_model.base.blocks)
    assert data['attention'].shape==(8,4,3,4,16)
    tolerance=0.01 if amp_dtype else 1e-6
    assert torch.allclose(data['attention'].float().sum(-1),torch.ones(8,4,3,4,device=device),atol=tolerance)
    ratio=data['delta'].float().norm(dim=-1)/data['global_feature'].float().norm(dim=-1)
    assert torch.allclose(ratio,torch.full_like(ratio,0.1),atol=tolerance)
    assert all(len(block._forward_hooks)==0 for block in model.global_model.base.blocks)
    saved={k:v.clone() for k,v in model.state_dict().items()}
    replica=JointLocalReID(TinyGlobal(),4,dim=16,heads=4,queries=4).to(device)
    replica.load_state_dict(saved,strict=True);model.eval();replica.eval()
    with torch.no_grad():assert torch.allclose(model(x),replica(x),atol=1e-6)
    return {'joint_gradients':True,'bn_updates':True,'multi_query_shape':True,'checkpoint_roundtrip':True}


def main():
    torch.manual_seed(1234);torch.set_num_threads(1)
    report={'cpu_fp32':verify_device(torch.device('cpu')),
            'cpu_bfloat16':verify_device(torch.device('cpu'),torch.bfloat16)}
    if torch.cuda.is_available():
        report['cuda_float16']=verify_device(torch.device('cuda:0'),torch.float16)
    # Zero patch value output must not leave a query residual in the descriptor.
    block=BoQBlock(16,4,4).eval()
    with torch.no_grad():
        block.cross_attention.in_proj_weight[32:].zero_()
        block.cross_attention.in_proj_bias[32:].zero_()
        block.cross_attention.out_proj.bias.zero_()
        output,_=block(torch.randn(2,16,16))
        assert torch.equal(output,torch.zeros_like(output))
    g=torch.randn(3,32,requires_grad=True);a=torch.randn(3,32,requires_grad=True)
    _,delta=fuse_local(g,a);norm_grad=torch.autograd.grad(delta.square().sum(),g)[0]
    assert torch.allclose(norm_grad,0.02*g,rtol=1e-5,atol=1e-6)
    report.update(query_residual_to_output_absent=True,differentiable_global_norm=True,
                  scope='Small substitute backbone only; full ViT is validated by the real training start')
    # Exercise the real metric/diagnostic code with the new five-dimensional maps.
    from types import SimpleNamespace
    from .processor import evaluate
    model=JointLocalReID(TinyGlobal(),4,dim=16,heads=4,queries=4).eval()
    images=torch.randn(16,3,16,16);pids=torch.arange(4).repeat(4)
    cams=torch.cat((torch.zeros(4,dtype=torch.long),torch.ones(12,dtype=torch.long)))
    loader=[(images[i:i+4],pids[i:i+4],cams[i:i+4],cams[i:i+4],torch.zeros(4,dtype=torch.long),None) for i in range(0,16,4)]
    cfg=SimpleNamespace(TEST=SimpleNamespace(FEAT_NORM='yes'))
    metrics,diagnostics=evaluate(cfg,model,loader,4,torch.device('cpu'))
    assert all(0<=v<=1 for v in metrics.values())
    assert 0<=diagnostics['mean_query_map_squared_cosine_overlap']<=1.000001
    report['five_dimensional_attention_evaluation']=True
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':main()
