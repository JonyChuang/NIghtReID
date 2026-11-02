"""Analytic retrieval loss, global memory and real-model checks; no training."""
import argparse
import inspect
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from .shared_cnl import SharedLayerCNL, create_shared_memory
from .model.reference_space import shared_bn_references


def vector(cosine):
    return torch.tensor([cosine, math.sqrt(max(0., 1-cosine*cosine)), 0.])


def fixture():
    memory = SharedLayerCNL([0,1,1,1,0], dim=3, temperature=.2)
    scores = torch.tensor([[.9,.1,.1,.1],[.1,.5,.1,.1],[-.8,-.8,-.8,-.8],[1.,1.,1.,1.]])
    keys = torch.stack([torch.stack([vector(float(s)) for s in row]) for row in scores])
    fused_keys = torch.stack([vector(s) for s in [.8,-.2,.6,1.]])
    memory.update(fused_keys, keys, torch.arange(1,5), torch.tensor([1,1,1,0]))
    return memory, vector(1.).reshape(1,3), vector(1.).repeat(1,4,1), torch.tensor([0]), torch.tensor([0])


class SharedCNLChecks(unittest.TestCase):
    def test_analytic_final_retrieval_loss_and_detached_mining(self):
        memory, fused, layers, indices, cameras = fixture()
        fused.requires_grad_();layers.requires_grad_()
        before = {k:v.clone() for k,v in memory.state_dict().items()}
        loss, stats = memory(fused, layers, indices, cameras)
        # Both positives and the negative are outside this one-image batch.
        # Fused scores deliberately differ from all four mining score vectors.
        expected = -torch.log_softmax(torch.tensor([.8,-.2,.6])/.2,dim=0)[:2].mean()
        self.assertAlmostEqual(float(loss), float(expected), places=5)
        self.assertEqual(stats['selected_positives'],2)
        self.assertEqual(stats['eligible_pairs'],3)
        self.assertEqual(stats['support_3'],2)
        self.assertEqual(stats['out_of_batch_positives'],2)
        self.assertEqual(stats['out_of_batch_candidates'],4)
        loss.backward()
        self.assertGreater(float(fused.grad.norm()),0.)
        self.assertIsNone(layers.grad)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in memory.state_dict().items()))
        self.assertEqual(sum(1 for _ in memory.parameters()),0)

    def test_out_of_batch_positive_and_negative_fused_features_affect_loss(self):
        memory,*args=fixture()
        original,stats=memory(*args)
        memory.fused_features[2]=vector(.7)
        better_positive,after=memory(*args)
        self.assertLess(float(better_positive),float(original))
        self.assertEqual(stats,after)
        memory.fused_features[3]=vector(-.8)
        better_negative,after=memory(*args)
        self.assertLess(float(better_negative),float(better_positive))
        self.assertEqual(stats,after)

    def test_no_cross_layer_cosine_or_global_threshold(self):
        memory,fused,layers,indices,cameras=fixture()
        _,stats=memory(fused,layers,indices,cameras)
        # .5 qualifies against layer10's .5 maximum, despite .9 in layer9.
        self.assertEqual(stats['discovered_layer_10'],1)
        self.assertEqual(stats['selected_positives'],2)
        layers[:,0]=torch.tensor([0.,0.,1.])
        loss,stats=memory(fused,layers,indices,cameras)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats['discovered_layer_9'],3)

    def test_threshold_includes_tied_maxima(self):
        memory=SharedLayerCNL([0,1,1,1],dim=3,gamma=1.)
        fused=torch.stack([vector(.2),vector(.3),vector(-.2)])
        layers=torch.stack([vector(.8).repeat(4,1),vector(.8).repeat(4,1),vector(-.5).repeat(4,1)])
        memory.update(fused,layers,torch.tensor([1,2,3]),torch.ones(3,dtype=torch.long))
        _,stats=memory(vector(1.).reshape(1,3),vector(1.).repeat(1,4,1),torch.tensor([0]),torch.tensor([0]))
        self.assertEqual(stats['selected_positives'],2)

    def test_empty_and_all_positive_cases_are_finite_differentiable_zero(self):
        for cameras in ([0],[0,0],[0,1]):
            memory=SharedLayerCNL(cameras,dim=3)
            fused=vector(1.).repeat(len(cameras),1).requires_grad_()
            layers=vector(1.).repeat(len(cameras),4,1)
            loss,stats=memory(fused,layers,torch.arange(len(cameras)),torch.tensor(cameras))
            self.assertEqual(float(loss),0.)
            self.assertEqual(stats['active_anchors'],0)
            loss.backward();self.assertTrue(torch.isfinite(fused.grad).all())
            self.assertFalse(memory.valid.any())

    def test_current_duplicate_images_replace_both_stale_entries(self):
        memory=SharedLayerCNL([0,1,1],dim=3,temperature=.2)
        stale=torch.stack([vector(-1.),vector(-.5)])
        memory.update(stale,stale[:,None].repeat(1,4,1),torch.tensor([1,2]),torch.tensor([1,1]))
        fused=vector(1.).repeat(3,1).requires_grad_()
        layers=vector(1.).repeat(3,4,1)
        loss,stats=memory(fused,layers,torch.tensor([0,1,1]),torch.tensor([0,1,1]))
        expected=-torch.log_softmax(torch.tensor([1.,-.5])/.2,dim=0)[0]
        self.assertAlmostEqual(float(loss),float(expected),places=5)
        self.assertEqual(stats['candidate_images'],3)
        self.assertEqual(stats['active_anchors'],1)
        self.assertTrue(torch.equal(memory.features[1],vector(-1.).repeat(4,1)))
        self.assertTrue(torch.equal(memory.fused_features[1],vector(-1.)))
        gradient=torch.autograd.grad(loss,fused)[0]
        self.assertTrue(torch.equal(gradient[1:],torch.zeros_like(gradient[1:])))

    def test_synchronized_five_key_momentum_and_duplicate_order(self):
        a=SharedLayerCNL([0,1],dim=3);b=SharedLayerCNL([0,1],dim=3)
        layers=torch.stack([vector(.2).repeat(4,1),vector(.6).repeat(4,1)])
        fused=torch.stack([vector(-.4),vector(.1)])
        a.update(fused,layers,torch.tensor([1,1]),torch.tensor([1,1]))
        b.update(fused.flip(0),layers.flip(0),torch.tensor([1,1]),torch.tensor([1,1]))
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        expected_layers=F.normalize(.2*a.features[1]+.8*vector(-.4).repeat(4,1),dim=-1)
        expected_fused=F.normalize(.2*a.fused_features[1]+.8*vector(.8),dim=-1)
        a.update(vector(.8).reshape(1,3),vector(-.4).repeat(1,4,1),torch.tensor([1]),torch.tensor([1]))
        self.assertTrue(torch.allclose(a.features[1],expected_layers))
        self.assertTrue(torch.allclose(a.fused_features[1],expected_fused))
        self.assertTrue(torch.allclose(a.features[1].norm(dim=-1),torch.ones(4)))
        self.assertAlmostEqual(float(a.fused_features[1].norm()),1.,places=6)
        self.assertEqual(int(a.valid.sum()),1)

    def test_bank_roundtrip_and_no_identity_argument_or_storage(self):
        memory,*args=fixture()
        restored=SharedLayerCNL([0,1,1,1,0],dim=3,temperature=.2)
        restored.load_state_dict(memory.state_dict(),strict=True)
        self.assertEqual(float(memory(*args)[0]),float(restored(*args)[0]))
        self.assertEqual(set(memory.state_dict()),{'cameras','features','fused_features','valid'})
        self.assertEqual(list(inspect.signature(memory.forward).parameters),['fused','layers','indices','cameras'])
        old={k:v for k,v in memory.state_dict().items() if k!='fused_features'}
        with self.assertRaises(RuntimeError):restored.load_state_dict(old,strict=True)
        class CameraOnlyRow:
            def __getitem__(self,index):
                if index!=2:raise AssertionError('CNL tried to read PID or another column')
                return 0
        cfg=SimpleNamespace(SOLVER=SimpleNamespace(CNL_WEIGHT=.1,CNL_GAMMA=.9,CNL_TEMPERATURE=.05,CNL_MOMENTUM=.2,CNL_CROSS_CAMERA=True))
        model=SimpleNamespace(base=SimpleNamespace(in_planes=3),mldc=SimpleNamespace(selected_layers=[9,10,11,12]))
        bank=create_shared_memory(cfg,model,SimpleNamespace(entries=[CameraOnlyRow()]),'cpu')
        self.assertEqual(tuple(bank.features.shape),(1,4,3))
        self.assertEqual(tuple(bank.fused_features.shape),(1,3))
        cfg.SOLVER.CNL_WEIGHT=0
        self.assertIsNone(create_shared_memory(cfg,model,None,'cpu'))

    def test_bn_mapping_matches_actual_fused_transform_without_extra_updates(self):
        torch.manual_seed(8)
        bn=torch.nn.BatchNorm1d(6).train()
        fused=torch.randn(4,6,requires_grad=True)
        neck=bn(fused)
        tracked=bn.num_batches_tracked.clone()
        reference=shared_bn_references(fused[:,None].expand(-1,4,-1),fused,bn)
        self.assertTrue(torch.allclose(reference[:,0],F.normalize(neck,dim=-1),atol=1e-6))
        self.assertFalse(reference.requires_grad)
        self.assertTrue(torch.equal(tracked,bn.num_batches_tracked))

    def test_invalid_slots_never_participate(self):
        memory,*args=fixture()
        memory.features[0]=vector(1.).repeat(4,1)
        memory.fused_features[0]=vector(1.)
        memory.valid[0]=True
        expected,stats=memory(*args)
        memory.features[3]=float('nan');memory.fused_features[3]=float('nan')
        memory.valid[3]=False
        loss,after=memory(*args)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(after['eligible_pairs'],2)
        self.assertEqual(after['active_anchors'],0)
        self.assertGreater(float(expected),0.)

    def test_loss_is_fp32_inside_autocast(self):
        memory,*args=fixture()
        expected,_=memory(*args)
        with torch.autocast('cpu',dtype=torch.bfloat16):actual,_=memory(*args)
        self.assertEqual(actual.dtype,torch.float32)
        self.assertTrue(torch.equal(expected,actual))


def full_model_check(config_name='night600_shared_cnl.yml'):
    from config import cfg as base
    from .model.make_model_mldc import MLDCReID,add_mldc_config
    from .model.backbones.vit_pytorch_mldc import TransReID
    from .loss.make_loss_mldc import make_loss
    from .processor.processor_mldc import save_checkpoint
    cfg=add_mldc_config(base.clone())
    cfg.merge_from_file(str(Path(__file__).parent/'configs'/config_name))
    assert cfg.SOLVER.AUXILIARY_LOSS=='shared_cnl' and cfg.SOLVER.PRESERVE_WEIGHT==0
    torch.manual_seed(1234)
    with patch.object(TransReID,'load_param',lambda *a,**k:None):model=MLDCReID(cfg,2,2,1)
    model.train();images=torch.randn(4,3,*cfg.INPUT.SIZE_TRAIN)
    labels=torch.tensor([0,0,1,1]);indices=torch.arange(4);cameras=torch.tensor([0,0,1,1])
    before=int(model.bottleneck.num_batches_tracked)
    scores,raw,unit,layers=model(images,return_cnl_features=True)
    assert not layers.requires_grad and layers.shape==(4,4,768)
    assert unit.requires_grad and int(model.bottleneck.num_batches_tracked)==before+1
    memory=SharedLayerCNL([0,0,1,1,1,1],dim=768,
        selection=cfg.SOLVER.CNL_SELECTION,min_support=cfg.SOLVER.CNL_MIN_SUPPORT,
        uncertain_gamma=cfg.SOLVER.CNL_UNCERTAIN_GAMMA)
    q=layers[0];direction=q.roll(1,1)
    direction=F.normalize(direction-(direction*q).sum(-1,keepdim=True)*q,dim=-1)
    layer_keys=torch.stack([.95*q+math.sqrt(1-.95**2)*direction,.5*q+math.sqrt(.75)*direction])
    fused_keys=torch.stack([unit[0].detach().roll(17),unit[0].detach().roll(35)])
    if cfg.SOLVER.CNL_SELECTION == 'fused_consensus':
        fq=unit[0].detach();fd=fq.roll(17)
        fd=F.normalize(fd-(fd*fq).sum()*fq,dim=-1)
        fused_keys=torch.stack([.95*fq+math.sqrt(1-.95**2)*fd,.4*fq+math.sqrt(.84)*fd])
    memory.update(fused_keys,layer_keys,torch.tensor([4,5]),torch.tensor([1,1]))
    before_memory={k:v.clone() for k,v in memory.state_dict().items()}
    cnl,stats=memory(unit,layers,indices,cameras)
    assert stats['active_anchors']>0 and stats['out_of_batch_positives']>0 and cnl>0
    targets=[model.mldc.layer_weights,model.bottleneck.weight]
    targets.extend(model.base.blocks[i].attn.qkv.weight for i in range(8,12))
    targets.append(model.classifier.weight)
    gradients=torch.autograd.grad(cnl,targets,retain_graph=True,allow_unused=True)
    for gradient in gradients[:-1]:
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum()>0
    assert gradients[-1] is None
    baseline,_=make_loss(cfg,2);total=baseline(scores,raw,labels)+cfg.SOLVER.CNL_WEIGHT*cnl
    total.backward()
    for prefix in ('base.blocks.','mldc.','bottleneck.','classifier.'):
        assert sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None)>0,prefix
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.equal(v,before_memory[k]) for k,v in memory.state_dict().items())
    class CheckpointStub(torch.nn.Module):
        def __init__(self):
            super().__init__();self.mldc=model.mldc;self.readout=model.readout;self.train_feature=model.train_feature
    stream=io.BytesIO();save_checkpoint(CheckpointStub(),cfg,0,stream,memory=memory)
    stream.seek(0);saved=torch.load(stream,weights_only=False)
    expected_variant=('mldc_fused_consensus_cnl_20261022' if cfg.SOLVER.CNL_SELECTION=='fused_consensus'
                      else 'mldc_multilayer_mined_retrieval_cnl_20261021')
    assert saved['variant']==expected_variant and 'preservation_memory' not in saved
    if cfg.SOLVER.CNL_SELECTION == 'fused_consensus':
        assert saved['shared_cnl_spec']['uncertain_denominator_excluded']
        assert saved['shared_cnl_spec']['min_support']==3 and saved['shared_cnl_spec']['uncertain_gamma']==.8
    assert saved['shared_cnl_spec']['contrast_space']=='fused_bn_after'
    assert not saved['shared_cnl_spec']['identity_labels_used']
    restored=SharedLayerCNL([0,0,1,1,1,1],dim=768,
        selection=cfg.SOLVER.CNL_SELECTION,min_support=cfg.SOLVER.CNL_MIN_SUPPORT,
        uncertain_gamma=cfg.SOLVER.CNL_UNCERTAIN_GAMMA)
    restored.load_state_dict(saved['shared_cnl_memory'],strict=True)
    assert all(torch.equal(v,memory.state_dict()[k]) for k,v in restored.state_dict().items())
    model.eval()
    with torch.no_grad():descriptor=model(images[:1])
    assert descriptor.shape==(1,768) and torch.isfinite(descriptor).all()
    return dict(full_vit_forward_backward=True,image_shape=list(images.shape),descriptor_dim=768,
        selection=cfg.SOLVER.CNL_SELECTION,min_support=cfg.SOLVER.CNL_MIN_SUPPORT,
        uncertain_gamma=cfg.SOLVER.CNL_UNCERTAIN_GAMMA,
        cnl_fusion_logit_gradient=gradients[0].tolist(),cnl_bn_gradient_norm=float(gradients[1].norm()),
        cnl_block9_12_gradient_norms=[float(g.norm()) for g in gradients[2:6]],
        learnable_fusion_retained=True,cnl_directly_supervises_final_retrieval=True,
        mining_references_detached=True,out_of_batch_positives=stats['out_of_batch_positives'],
        bn_updates_per_training_forward=1,shared_cnl_checkpoint_roundtrip=True,
        memory_unchanged_by_forward_backward=True,ssl_loaded=False,dataset_evaluated=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-model',action='store_true')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args();torch.set_num_threads(1)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SharedCNLChecks))
    if not result.wasSuccessful():raise SystemExit(1)
    report=dict(all_passed=True,unit_checks=result.testsRun,optimizer_steps=0,training_started=False)
    if args.full_model:report.update(full_model_check())
    if args.report:args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
