"""Consensus/uncertain-mask checks. Full-model option never takes optimizer steps."""
import argparse
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from torch.nn import functional as F
from .shared_cnl import SharedLayerCNL,create_shared_memory


def vector(cosine):
    return torch.tensor([cosine,math.sqrt(max(0.,1-cosine*cosine)),0.])


def fixture():
    bank=SharedLayerCNL([0,1,1,1,1,1,1,0],dim=3,temperature=.2,selection='fused_consensus')
    cosines=[[.86,.86,.86,.1],[.95,.1,.1,.1],[.1,.1,.1,.95],
             [.3,.3,.3,.3],[.82,.1,.1,.1],[-.4,-.4,-.4,-.4],[1.,1.,1.,1.]]
    layers=torch.stack([torch.stack([vector(c) for c in row]) for row in cosines])
    fused=torch.stack([vector(c) for c in [.87,.96,.2,.82,.2,.5,1.]])
    bank.update(fused,layers,torch.arange(1,8),torch.tensor([1,1,1,1,1,1,0]))
    return bank,vector(1.).reshape(1,3),vector(1.).repeat(1,4,1),torch.tensor([0]),torch.tensor([0])


class ConsensusChecks(unittest.TestCase):
    def test_analytic_loss_excludes_both_layer_and_fused_uncertainty(self):
        bank,fused,layers,indices,cameras=fixture()
        fused.requires_grad_();layers.requires_grad_()
        before={k:v.clone() for k,v in bank.state_dict().items()}
        loss,s=bank(fused,layers,indices,cameras)
        # Only positive .87 and negative .50 remain. The harder .96 key and
        # fused-only .82 and layer-only candidates are uncertain, not negatives.
        expected=torch.logsumexp(torch.tensor([.87,.5])/.2,dim=0)-.87/.2
        self.assertAlmostEqual(float(loss),float(expected),places=5)
        self.assertEqual((s['selected_positives'],s['uncertain_pairs'],s['negative_pairs']),(1,4,1))
        self.assertEqual(s['eligible_pairs'],6);self.assertEqual(s['denominator_pairs'],2)
        self.assertEqual(s['support_3'],1);self.assertEqual(s['support_1'],0)
        loss.backward()
        self.assertGreater(float(fused.grad.norm()),0.);self.assertIsNone(layers.grad)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in bank.state_dict().items()))

    def test_uncertain_key_score_does_not_affect_loss_but_negative_does(self):
        bank,*args=fixture();original,stats=bank(*args)
        bank.fused_features[4]=vector(.85)
        same,after=bank(*args)
        self.assertTrue(torch.equal(original,same));self.assertEqual(stats,after)
        bank.fused_features[6]=vector(.7)  # Below .8 * .96, still a negative.
        harder,after=bank(*args)
        self.assertGreater(float(harder),float(original));self.assertEqual(stats,after)

    def test_three_layer_confirmation_and_fused_threshold_both_required(self):
        bank,*args=fixture()
        bank.fused_features[1]=vector(.7)  # Three layers alone cannot nominate.
        zero,s=bank(*args);self.assertEqual(float(zero),0.);self.assertEqual(s['selected_positives'],0)
        bank,*args=fixture()
        bank.features[2]=torch.stack([vector(.95),vector(.95),vector(.1),vector(.1)])
        _,s=bank(*args);self.assertEqual(s['selected_positives'],1)
        bank.features[2,2]=vector(.95)
        _,s=bank(*args);self.assertEqual(s['selected_positives'],2)

    def test_threshold_equality_at_point9_and_point8(self):
        bank=SharedLayerCNL([0,1,1,1,1],dim=3,selection='fused_consensus')
        fused=torch.stack([vector(c) for c in [.9,1.,.8,.5]])
        layers=torch.stack([vector(c).repeat(4,1) for c in [.9,1.,.1,-.5]])
        bank.update(fused,layers,torch.arange(1,5),torch.ones(4,dtype=torch.long))
        _,s=bank(vector(1.).reshape(1,3),vector(1.).repeat(1,4,1),torch.tensor([0]),torch.tensor([0]))
        self.assertEqual((s['selected_positives'],s['uncertain_pairs'],s['negative_pairs']),(2,1,1))

    def test_nonpositive_maxima_do_not_force_neighbors(self):
        for cosine in [-.5,0.]:
            bank=SharedLayerCNL([0,1,1],dim=3,selection='fused_consensus')
            bank.update(vector(cosine).repeat(2,1),vector(cosine).repeat(2,4,1),
                        torch.tensor([1,2]),torch.ones(2,dtype=torch.long))
            q=vector(1.).reshape(1,3).requires_grad_()
            loss,s=bank(q,vector(1.).repeat(1,4,1),torch.tensor([0]),torch.tensor([0]))
            self.assertEqual(float(loss),0.);self.assertEqual(s['selected_positives'],0)
            self.assertEqual(s['uncertain_pairs'],0);loss.backward()
            self.assertTrue(torch.isfinite(q.grad).all())

    def test_empty_or_no_negative_batch_returns_differentiable_zero(self):
        for cameras in [[0],[0,0],[0,1]]:
            bank=SharedLayerCNL(cameras,dim=3,selection='fused_consensus')
            q=vector(1.).repeat(len(cameras),1).requires_grad_()
            loss,s=bank(q,vector(1.).repeat(len(cameras),4,1),torch.arange(len(cameras)),torch.tensor(cameras))
            self.assertEqual(float(loss),0.);self.assertEqual(s['active_anchors'],0)
            loss.backward();self.assertTrue(torch.isfinite(q.grad).all());self.assertFalse(bank.valid.any())

    def test_duplicate_query_is_one_candidate_and_no_memory_writes(self):
        bank,fused,layers,indices,cameras=fixture()
        before={k:v.clone() for k,v in bank.state_dict().items()}
        loss,s=bank(fused.repeat(2,1),layers.repeat(2,1,1),indices.repeat(2),cameras.repeat(2))
        self.assertTrue(torch.isfinite(loss));self.assertEqual(s['candidate_images'],8)
        self.assertEqual(s['selected_positives'],2);self.assertEqual(s['eligible_pairs'],12)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in bank.state_dict().items()))

    def test_factory_no_pid_and_explicit_selection(self):
        class CameraOnly:
            def __getitem__(self,index):
                if index!=2:raise AssertionError('PID access from CNL factory')
                return 0
        cfg=SimpleNamespace(SOLVER=SimpleNamespace(CNL_WEIGHT=.1,CNL_GAMMA=.9,CNL_TEMPERATURE=.05,
            CNL_MOMENTUM=.2,CNL_CROSS_CAMERA=True,CNL_SELECTION='fused_consensus',CNL_MIN_SUPPORT=3,CNL_UNCERTAIN_GAMMA=.8))
        model=SimpleNamespace(base=SimpleNamespace(in_planes=3),mldc=SimpleNamespace(selected_layers=[9,10,11,12]))
        bank=create_shared_memory(cfg,model,SimpleNamespace(entries=[CameraOnly()]),'cpu')
        self.assertEqual(bank.selection,'fused_consensus');self.assertEqual(bank.min_support,3)
        self.assertEqual(list(inspect.signature(bank.forward).parameters),['fused','layers','indices','cameras'])
        self.assertEqual(set(bank.state_dict()),{'cameras','features','fused_features','valid'})

    def test_invalid_threshold_support_or_selection_rejected(self):
        for options in [dict(uncertain_gamma=.95),dict(min_support=0),dict(min_support=5),
                        dict(selection='unknown'),dict(uncertain_gamma=float('nan'))]:
            with self.assertRaises(ValueError):SharedLayerCNL([0,1],dim=3,**options)

    def test_fp32_loss_under_autocast(self):
        bank,*args=fixture();expected,_=bank(*args)
        with torch.autocast('cpu',dtype=torch.bfloat16):actual,_=bank(*args)
        self.assertEqual(actual.dtype,torch.float32);self.assertTrue(torch.equal(actual,expected))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-model',action='store_true');parser.add_argument('--report',type=Path)
    args=parser.parse_args();torch.set_num_threads(1)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ConsensusChecks))
    if not result.wasSuccessful():raise SystemExit(1)
    report=dict(all_passed=True,unit_checks=result.testsRun,optimizer_steps=0,training_started=False)
    if args.full_model:
        from .verify_shared_cnl import full_model_check
        report.update(full_model_check('night600_shared_cnl_consensus.yml'))
    if args.report:args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
