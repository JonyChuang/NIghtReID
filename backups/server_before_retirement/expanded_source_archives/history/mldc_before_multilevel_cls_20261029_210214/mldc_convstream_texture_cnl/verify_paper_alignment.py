"""Focused regression checks for PAD AMU, BN, weighting and KL gradient paths."""
import math
from pathlib import Path
import torch
from torch.nn import functional as F
import yaml
from yacs.config import CfgNode
from .prototype_memory import PrototypeMemory
from .model.prototype_attention import PrototypeAttentionHead,attention_distillation

def check_paper_alignment(model):
    device=next(model.parameters()).device
    def direction(degrees):
        angle=math.radians(degrees)
        return torch.tensor([math.cos(angle),math.sin(angle)],device=device)
    old=direction(0);negative_near_old=direction(10);negative_near_sample=direction(-20)
    sample=direction(-10)
    bank=PrototypeMemory([0,1,2],3,2,amu_temperature=.05).to(device)
    bank.features.copy_(torch.stack([old,negative_near_old,negative_near_sample]));bank.valid.fill_(True)
    assert old@negative_near_old>old@negative_near_sample
    assert sample@negative_near_sample>sample@negative_near_old
    expected_mu=torch.sigmoid((old@sample-negative_near_old@sample)/.05)
    expected=F.normalize((1-expected_mu)*old+expected_mu*sample,dim=0)
    stats=bank.update(sample[None],torch.tensor([0],device=device),torch.tensor([0],device=device))
    assert abs(stats['amu_mu_sum']-float(expected_mu))<1e-6
    assert torch.allclose(bank.features[0],expected,atol=1e-7)
    assert bank.image_counts.tolist()==[1,0,0]
    assert torch.equal(bank.features[1],negative_near_old)
    duplicate=PrototypeMemory([0,1,2],3,2,amu_temperature=.05).to(device)
    duplicate.features.copy_(torch.stack([old,negative_near_old,negative_near_sample]));duplicate.valid.fill_(True)
    duplicate.update(sample.expand(2,-1),torch.tensor([0,0],device=device),torch.tensor([0,0],device=device))
    assert torch.allclose(duplicate.features,bank.features)
    assert duplicate.updates.tolist()==[1,0,0] and duplicate.image_counts.tolist()==[1,0,0]
    unfavorable=torch.tensor([0.,1.],device=device)
    low_mu=torch.sigmoid((old@unfavorable-negative_near_old@unfavorable)/.05)
    assert low_mu<expected_mu
    # Two different images of one identity must both contribute, not just a mean.
    sequential=PrototypeMemory([0,0,1,2],3,2,amu_temperature=.05).to(device)
    sequential.features.copy_(torch.stack([old,negative_near_old,negative_near_sample]));sequential.valid.fill_(True)
    second=direction(-5)
    hard2=negative_near_old if expected@negative_near_old>=expected@negative_near_sample else negative_near_sample
    mu2=torch.sigmoid((expected@second-hard2@second)/.05)
    expected2=F.normalize((1-mu2)*expected+mu2*second,dim=0)
    sequential.update(torch.stack([sample,second]),torch.tensor([0,0],device=device),torch.tensor([0,1],device=device))
    assert torch.allclose(sequential.features[0],expected2,atol=1e-7)
    assert sequential.updates[0]==2 and sequential.image_counts[0]==2
    # A new class has no previous prototype; initialize it without a fake negative.
    fresh=PrototypeMemory([0],2,2).to(device)
    fs=fresh.update(sample[None],torch.tensor([0],device=device),torch.tensor([0],device=device))
    assert torch.allclose(fresh.features[0],F.normalize(sample,dim=0),atol=1e-7) and fs['bootstrap_samples']==1
    assert torch.isfinite(fresh.features).all()
    epoch=model.pad_settings.WARMUP_EPOCHS+1
    terms=dict(prototype=torch.tensor(.7,device=device),attention=torch.tensor(.2,device=device))
    assert model.pad_settings.ATT_WEIGHT==5 and model.pad_settings.RAMP_EPOCHS==0
    assert model.attention_weight(epoch)==5 and model.attention_weight(120)==5
    assert model.attention_weight(model.pad_settings.WARMUP_EPOCHS)==0
    assert torch.allclose(model.distillation_loss(terms,epoch),torch.tensor(1.7,device=device))
    # Merge the actual shipped YAMLs so defaults cannot silently mask old values.
    from .model.make_model_mldc import add_mldc_config
    from config import cfg as defaults
    configs={}
    for name in ['pad_p12_nightreid.yml','pad_prototype_no_distill_nightreid.yml','pad_head_only_nightreid.yml','baseline_nightreid.yml']:
        cfg=add_mldc_config(defaults.clone());cfg.merge_from_file(str(Path(__file__).parent/'configs'/name))
        assert cfg.PAD.UPDATE_MODE=='adaptive' and cfg.PAD.AMU_TEMPERATURE==.05
        assert cfg.PAD.EMBED_NORM=='batchnorm' and cfg.PAD.RAMP_EPOCHS==0
        assert cfg.PAD.PROTO_DIM==768 and cfg.PAD.STUDENT_FEATURE=='fused'
        assert cfg.PAD.ATT_WEIGHT==(0. if 'no_distill' in name or 'head_only' in name else 5.)
        configs[name]=dict(weight=cfg.PAD.ATT_WEIGHT,update=cfg.PAD.UPDATE_MODE,norm=cfg.PAD.EMBED_NORM)
    return dict(amu_negative_selected_by_prototype=True,amu_not_selected_by_current_sample=True,
        amu_mu=float(expected_mu),amu_low_quality_mu=float(low_mu),amu_exact_formula=True,
        duplicate_augmentations_coalesced=True,distinct_images_updated_sequentially=True,
        bootstrap_without_valid_negative=True,constant_attention_weight=5.,
        shipped_configurations=configs)
