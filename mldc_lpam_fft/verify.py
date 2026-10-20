"""Integration checks only. No dataset, pretraining checkpoint or optimizer step."""
import copy
import gc
import json
from pathlib import Path


def main():
    import torch
    from torch import nn
    import torch.nn.functional as F
    from config import cfg as base_cfg
    from .model import add_config, LPAMFFTReID, fixed_equal_global, objectives
    from .lpam import SelfAttention
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    cfg=add_config(base_cfg.clone());cfg.merge_from_file(str(Path(__file__).with_name('variants/fixed_global_lpam_fft_concat/config.yml')));cfg.freeze()
    before=torch.random.get_rng_state()
    x=torch.randn(12,4,768,requires_grad=True)
    average=fixed_equal_global(x)
    assert torch.equal(average,x[8:12].mean(0))
    average.sum().backward()
    assert torch.equal(x.grad[:8],torch.zeros_like(x.grad[:8]))
    assert torch.equal(x.grad[8:],torch.full_like(x.grad[8:],.25))
    # Compare the exact FFT attention with spatial dot-product attention using
    # identical weights. At N=1 both are V times the learned head weight.
    attention=SelfAttention(768,heads=8)
    inp=torch.randn(4,1,768,requires_grad=True)
    actual=attention(inp)
    values=attention.value(inp).view(4,1,8,96).transpose(1,2)
    reference=(values*attention.weights).transpose(1,2).contiguous().view(4,1,768)
    difference=(actual-reference).abs().max().item()
    assert torch.allclose(actual,reference,atol=1e-7,rtol=1e-6)
    actual.square().mean().backward()
    q_grad=attention.query.weight.grad.abs().max().item()
    k_grad=attention.key.weight.grad.abs().max().item()
    v_grad=attention.value.weight.grad.norm().item()
    assert q_grad==k_grad==0 and v_grad>0
    del attention,actual,reference,values,inp,x,average
    gc.collect()
    # Instantiate the full real ViT architecture, without loading server data.
    model=LPAMFFTReID(cfg,8,0,0,initialize=False)
    assert all(p.requires_grad for b in model.base.blocks for p in b.parameters())
    assert not any('layer_weights' in n for n,_ in model.named_parameters())
    assert torch.equal(model.readout_weights(),torch.tensor([0.]*8+[.25]*4))
    model.train()
    layers=torch.randn(12,4,768,requires_grad=True)
    data=model.from_layers(layers)
    labels=torch.tensor([0,0,1,1])
    loss,ids,tri=objectives(data,labels,cfg)
    loss.backward()
    assert data['retrieval'].shape==(4,3072)
    assert layers.grad[:8].abs().sum()>0 and layers.grad[8:].abs().sum()>0
    assert model.lpam.conv_block.conv1.weight.grad.abs().sum()>0
    assert model.lpam.conv_block.self_attn.value.weight.grad.abs().sum()>0
    model.zero_grad(set_to_none=True)
    del layers,data,loss,ids,tri
    gc.collect()
    model.eval()
    with torch.inference_mode():
        full=model(torch.randn(1,3,256,128),diagnostics=True)
        assert full['layers'].shape==(12,1,768)
        assert full['retrieval'].shape==(1,3072)
        assert torch.equal(full['raw_global'],full['layers'][8:12].mean(0))
    del full,model
    gc.collect()
    base=cfg.clone();base.defrost();base.MODEL.LPAM_ENABLED=False;base.freeze()
    control=LPAMFFTReID(base,8,0,0,initialize=False).eval()
    with torch.no_grad():
        parts=control.from_layers(torch.randn(12,2,768))
        assert parts['retrieval'].shape==(2,768) and torch.equal(parts['retrieval'],parts['global'])
    result=dict(global_mean_exact=True,global_gradient_each=.25,global_weights_trainable=False,
                active_backbone_trainable=True,full_vit_forward_shape=[1,3072],
                cls_pyramid_backward_passed=True,baseline_descriptor_dim=768,
                fft_attention_vs_value_max_abs_difference=difference,
                fft_query_gradient_max=q_grad,fft_key_gradient_max=k_grad,value_gradient_norm=v_grad,
                conclusion='N=1 makes FFT attention equivalent to V projection with head scaling; it does not add spatial selection',
                pretrained_checkpoint_loaded=False,dataset_evaluated=False,optimizer_steps=0)
    print('V4_VERIFICATION',json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    main()
