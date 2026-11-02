"""Analytic loss, memory and real-model checks, with zero optimizer steps."""
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
from .model.reference_space import shared_bn_layer_queries, shared_bn_references


def vector(cosine):
    return torch.tensor([cosine, math.sqrt(max(0., 1-cosine*cosine)), 0.])


def fixture():
    memory = SharedLayerCNL([0,1,1,1,0], dim=3, temperature=.2)
    scores = torch.tensor([[.9,.1,.1,.1],[.1,.5,.1,.1],[-.8,-.8,-.8,-.8],[1.,1.,1.,1.]])
    keys = torch.stack([torch.stack([vector(float(s)) for s in row]) for row in scores])
    memory.update(keys, torch.arange(1,5), torch.tensor([1,1,1,0]))
    return memory, vector(1.).repeat(1,4,1), torch.tensor([0]), torch.tensor([0]), scores[:3].T


class SharedCNLChecks(unittest.TestCase):
    def test_analytic_per_layer_threshold_union_and_all_positive_loss(self):
        memory, query, indices, cameras, scores = fixture()
        query.requires_grad_()
        before = {k:v.clone() for k,v in memory.state_dict().items()}
        loss, stats = memory(query, indices, cameras)
        expected = -torch.log_softmax(scores/.2,dim=-1)[:,:2].mean()
        self.assertAlmostEqual(float(loss), float(expected), places=5)
        self.assertEqual(stats['selected_positives'],2)
        self.assertEqual(stats['eligible_pairs'],3)
        self.assertEqual(stats['support_3'],2)
        loss.backward()
        self.assertTrue((query.grad.norm(dim=-1)>0).all())
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in memory.state_dict().items()))
        self.assertEqual(sum(1 for _ in memory.parameters()),0)

    def test_no_cross_layer_cosine_or_global_threshold(self):
        memory, query, indices, cameras, _ = fixture()
        _,stats=memory(query,indices,cameras)
        # .5 qualifies against layer10's .5 maximum, despite .9 in layer9.
        self.assertEqual(stats['discovered_layer_10'],1)
        self.assertEqual(stats['selected_positives'],2)
        # A high vector in another layer cannot raise layer9's maximum.
        query[:,0]=torch.tensor([0.,0.,1.])
        loss,stats=memory(query,indices,cameras)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats['discovered_layer_9'],3)  # All same-layer cosines are exactly zero.

    def test_threshold_includes_tied_maxima(self):
        memory=SharedLayerCNL([0,1,1,1],dim=3,gamma=1.)
        memory.update(torch.stack([vector(.8).repeat(4,1),vector(.8).repeat(4,1),vector(-.5).repeat(4,1)]),
                      torch.tensor([1,2,3]),torch.ones(3,dtype=torch.long))
        _,stats=memory(vector(1.).repeat(1,4,1),torch.tensor([0]),torch.tensor([0]))
        self.assertEqual(stats['selected_positives'],2)

    def test_empty_and_all_positive_cases_are_finite_differentiable_zero(self):
        for cameras in ([0],[0,0],[0,1]):
            memory=SharedLayerCNL(cameras,dim=3)
            query=vector(1.).repeat(len(cameras),4,1).requires_grad_()
            loss,stats=memory(query,torch.arange(len(cameras)),torch.tensor(cameras))
            self.assertEqual(float(loss),0.)
            self.assertEqual(stats['active_anchors'],0)
            loss.backward();self.assertTrue(torch.isfinite(query.grad).all())
            self.assertFalse(memory.valid.any())

    def test_current_duplicate_images_replace_stale_entries(self):
        memory=SharedLayerCNL([0,1,1],dim=3)
        memory.update(torch.stack([vector(-1.).repeat(4,1),vector(-.5).repeat(4,1)]),
                      torch.tensor([1,2]),torch.tensor([1,1]))
        query=vector(1.).repeat(3,4,1)
        loss,stats=memory(query,torch.tensor([0,1,1]),torch.tensor([0,1,1]))
        self.assertEqual(stats['candidate_images'],3)
        self.assertGreater(stats['active_anchors'],0)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.equal(memory.features[1],vector(-1.).repeat(4,1)))

    def test_atomic_four_layer_momentum_and_duplicate_order(self):
        a=SharedLayerCNL([0,1],dim=3);b=SharedLayerCNL([0,1],dim=3)
        keys=torch.stack([vector(.2).repeat(4,1),vector(.6).repeat(4,1)])
        a.update(keys,torch.tensor([1,1]),torch.tensor([1,1]))
        b.update(keys.flip(0),torch.tensor([1,1]),torch.tensor([1,1]))
        self.assertTrue(torch.equal(a.features,b.features))
        old=a.features[1].clone();new=vector(-.4).repeat(1,4,1)
        expected=F.normalize(.2*old+.8*new[0],dim=-1)
        a.update(new,torch.tensor([1]),torch.tensor([1]))
        self.assertTrue(torch.allclose(a.features[1],expected))
        self.assertTrue(torch.allclose(a.features[1].norm(dim=-1),torch.ones(4)))
        self.assertEqual(int(a.valid.sum()),1)

    def test_bank_roundtrip_and_no_identity_argument_or_storage(self):
        memory,*args=fixture()
        restored=SharedLayerCNL([0,1,1,1,0],dim=3,temperature=.2)
        restored.load_state_dict(memory.state_dict(),strict=True)
        self.assertEqual(float(memory(*args[:3])[0]),float(restored(*args[:3])[0]))
        self.assertEqual(set(memory.state_dict()),{'cameras','features','valid'})
        self.assertEqual(list(inspect.signature(memory.forward).parameters),['layers','indices','cameras'])
        class CameraOnlyRow:
            def __getitem__(self,index):
                if index!=2:raise AssertionError('CNL tried to read a PID or another column')
                return 0
        cfg=SimpleNamespace(SOLVER=SimpleNamespace(CNL_WEIGHT=.1,CNL_GAMMA=.9,CNL_TEMPERATURE=.05,CNL_MOMENTUM=.2,CNL_CROSS_CAMERA=True))
        model=SimpleNamespace(base=SimpleNamespace(in_planes=3),mldc=SimpleNamespace(selected_layers=[9,10,11,12]))
        bank=create_shared_memory(cfg,model,SimpleNamespace(entries=[CameraOnlyRow()]),'cpu')
        self.assertEqual(tuple(bank.features.shape),(1,4,3))
        cfg.SOLVER.CNL_WEIGHT=0
        self.assertIsNone(create_shared_memory(cfg,model,None,'cpu'))

    def test_bn_mapping_is_identical_but_queries_have_gradients(self):
        torch.manual_seed(8)
        bn=torch.nn.BatchNorm1d(6).train()
        fused=torch.randn(4,6,requires_grad=True)
        layers=torch.randn(4,4,6,requires_grad=True)
        bn(fused)
        tracked=bn.num_batches_tracked.clone()
        queries=shared_bn_layer_queries(layers,fused,bn)
        reference=shared_bn_references(layers,fused,bn)
        self.assertTrue(torch.equal(queries.detach(),reference))
        self.assertTrue(torch.equal(tracked,bn.num_batches_tracked))
        gradients=torch.autograd.grad(queries[:,:,0].sum(),[layers,bn.weight,fused],allow_unused=True)
        self.assertGreater(float(gradients[0].abs().sum()),0.)
        self.assertGreater(float(gradients[1].abs().sum()),0.)
        self.assertIsNone(gradients[2])

    def test_loss_is_fp32_inside_autocast(self):
        memory,query,indices,cameras,_=fixture()
        expected,_=memory(query,indices,cameras)
        with torch.autocast('cpu',dtype=torch.bfloat16):actual,_=memory(query,indices,cameras)
        self.assertEqual(actual.dtype,torch.float32)
        self.assertTrue(torch.equal(expected,actual))


def full_model_check():
    from config import cfg as base
    from .model.make_model_mldc import MLDCReID,add_mldc_config
    from .model.backbones.vit_pytorch_mldc import TransReID
    from .loss.make_loss_mldc import make_loss
    from .processor.processor_mldc import save_checkpoint
    cfg=add_mldc_config(base.clone())
    cfg.merge_from_file(str(Path(__file__).parent/'configs/night600_shared_cnl.yml'))
    assert cfg.SOLVER.AUXILIARY_LOSS=='shared_cnl' and cfg.SOLVER.PRESERVE_WEIGHT==0
    torch.manual_seed(1234)
    with patch.object(TransReID,'load_param',lambda *a,**k:None):model=MLDCReID(cfg,2,2,1)
    model.train();images=torch.randn(4,3,*cfg.INPUT.SIZE_TRAIN)
    labels=torch.tensor([0,0,1,1]);indices=torch.arange(4);cameras=torch.tensor([0,0,1,1])
    before=int(model.bottleneck.num_batches_tracked)
    scores,raw,unit,layers=model(images,return_cnl_features=True)
    assert layers.requires_grad and layers.shape==(4,4,768)
    assert int(model.bottleneck.num_batches_tracked)==before+1
    memory=SharedLayerCNL([0,0,1,1,1,1],dim=768)
    q=layers[0].detach();direction=q.roll(1,1)
    direction=F.normalize(direction-(direction*q).sum(-1,keepdim=True)*q,dim=-1)
    memory.update(torch.stack([.95*q+math.sqrt(1-.95**2)*direction,.5*q+math.sqrt(.75)*direction]),
                  torch.tensor([4,5]),torch.tensor([1,1]))
    before_memory={k:v.clone() for k,v in memory.state_dict().items()}
    cnl,stats=memory(layers,indices,cameras)
    assert stats['active_anchors']>0 and cnl>0
    gradient=torch.autograd.grad(cnl,layers,retain_graph=True)[0]
    assert torch.isfinite(gradient).all() and (gradient[0].norm(dim=-1)>0).all()
    assert torch.autograd.grad(cnl,model.mldc.layer_weights,retain_graph=True,allow_unused=True)[0] is None
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
    assert saved['variant']=='mldc_shared_layer_cnl_20261021' and 'preservation_memory' not in saved
    assert not saved['shared_cnl_spec']['identity_labels_used']
    restored=SharedLayerCNL([0,0,1,1,1,1],dim=768)
    restored.load_state_dict(saved['shared_cnl_memory'],strict=True)
    assert all(torch.equal(v,memory.state_dict()[k]) for k,v in restored.state_dict().items())
    model.eval()
    with torch.no_grad():descriptor=model(images[:1])
    assert descriptor.shape==(1,768) and torch.isfinite(descriptor).all()
    assert model.mldc.layer_weights.requires_grad
    return dict(full_vit_forward_backward=True,image_shape=list(images.shape),descriptor_dim=768,
        layer_query_gradient_norms=gradient[0].norm(dim=-1).tolist(),learnable_fusion_retained=True,
        fusion_logits_supervised_by_original_id_triplet=True,
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
