"""CNL start boundary, gradients, online memory and saved schedule; no optimizer."""
import argparse
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import torch

from .cnl_schedule import auxiliary_weight_for_epoch, auxiliary_loss_for_step
from .verify_consensus_cnl import fixture


def config():
    from config import cfg as base
    from .model.make_model_mldc import add_mldc_config
    cfg = add_mldc_config(base.clone())
    cfg.merge_from_file(str(Path(__file__).parent / 'configs/night600_shared_cnl_consensus_warmup30.yml'))
    cfg.freeze()
    return cfg


class WarmupChecks(unittest.TestCase):
    def test_exact_30_31_boundary_and_constant_afterward(self):
        cfg = config()
        self.assertTrue(all(auxiliary_weight_for_epoch(cfg, epoch) == 0. for epoch in range(1, 31)))
        self.assertTrue(all(auxiliary_weight_for_epoch(cfg, epoch) == .1 for epoch in range(31, 121)))

    def test_new_config_changes_only_warmup_and_output(self):
        from yacs.config import CfgNode
        cfg = config()
        old = CfgNode.load_cfg((Path(__file__).parent / 'configs/night600_shared_cnl_consensus.yml').read_text())
        new = CfgNode.load_cfg((Path(__file__).parent / 'configs/night600_shared_cnl_consensus_warmup30.yml').read_text())
        new.SOLVER.pop('CNL_WARMUP_EPOCHS')
        new.OUTPUT_DIR = old.OUTPUT_DIR
        self.assertEqual(new.dump(), old.dump())
        self.assertEqual(cfg.MODEL.PRETRAIN_CHOICE, 'imagenet')
        self.assertEqual(cfg.SOLVER.MAX_EPOCHS, 120)
        self.assertEqual(cfg.SOLVER.WARMUP_EPOCHS, 20)  # Original LR warmup is independent.
        self.assertEqual(cfg.SOLVER.MLDC_WEIGHT_LR_FACTOR, 1.)

    def test_warmup_does_not_mine_or_change_primary_gradients(self):
        bank, unit, layers, indices, cameras = fixture()
        memory = Mock(side_effect=AssertionError('CNL must not execute during warmup'))
        unit.requires_grad_()
        original = (unit - .3).square().mean()
        expected = torch.autograd.grad(original, unit, retain_graph=True)[0]
        weight = auxiliary_weight_for_epoch(config(), 30)
        auxiliary, stats = auxiliary_loss_for_step(memory, weight, unit, layers, indices, cameras)
        loss = original + weight * auxiliary
        loss.backward()
        self.assertTrue(torch.equal(loss, original))
        self.assertTrue(torch.equal(unit.grad, expected))
        self.assertEqual(stats, {})
        self.assertFalse(auxiliary.requires_grad)
        memory.assert_not_called()

    def test_epoch31_applies_exact_point1_cnl_gradient(self):
        bank, unit, layers, indices, cameras = fixture()
        unit.requires_grad_()
        original = (unit - .3).square().mean()
        direct, _ = bank(unit, layers, indices, cameras)
        expected = torch.autograd.grad(original + .1 * direct, unit, retain_graph=True)[0]
        before = {k: v.clone() for k, v in bank.state_dict().items()}
        weight = auxiliary_weight_for_epoch(config(), 31)
        auxiliary, stats = auxiliary_loss_for_step(bank, weight, unit, layers, indices, cameras)
        (original + weight * auxiliary).backward()
        self.assertTrue(torch.equal(unit.grad, expected))
        self.assertEqual(stats['selected_positives'], 1)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in bank.state_dict().items()))

    def test_warmup_memory_is_constructed_and_available_on_activation(self):
        from .shared_cnl import create_shared_memory
        cfg = config()
        model = SimpleNamespace(base=SimpleNamespace(in_planes=3), mldc=SimpleNamespace(selected_layers=[9,10,11,12]))
        dataset = SimpleNamespace(entries=[('a', None, 0), ('b', None, 1), ('c', None, 1)])
        bank = create_shared_memory(cfg, model, dataset, 'cpu')
        self.assertIsNotNone(bank)  # Configured weight is .1 even when the effective weight is zero.
        from .verify_consensus_cnl import vector
        historical_fused = torch.stack([vector(.95), vector(.2)])
        historical_layers = historical_fused[:,None,:].repeat(1,4,1)
        ids = torch.tensor([1,2]); cams = torch.tensor([1,1])
        # Buffer-only writes represent the unchanged successful-step update hook.
        bank.update(historical_fused, historical_layers, ids, cams)
        historical_fused[0] = vector(.9)
        bank.update(historical_fused, historical_layers, ids, cams)
        self.assertEqual(int(bank.valid.sum()), 2)
        q = vector(1.).reshape(1,3)
        loss, stats = auxiliary_loss_for_step(bank, auxiliary_weight_for_epoch(cfg,31),
            q, q[:,None,:].repeat(1,4,1), torch.tensor([0]), torch.tensor([0]))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats['out_of_batch_candidates'], 2)
        self.assertEqual(stats['out_of_batch_positives'], 1)

    def test_checkpoint_records_schedule_at_30_and_31(self):
        from .processor.processor_mldc import save_checkpoint
        cfg = config()
        model = torch.nn.Linear(3,3)
        model.mldc = SimpleNamespace(selected_layers=[9,10,11,12], temperature=.9)
        model.readout = 'mldc'
        bank, *_ = fixture()
        for epoch, expected in [(30,0.), (31,.1)]:
            stream = io.BytesIO()
            save_checkpoint(model,cfg,epoch,stream,memory=bank)
            stream.seek(0)
            saved = torch.load(stream,weights_only=False)
            spec = saved['shared_cnl_spec']
            self.assertEqual(spec['warmup_epochs'],30)
            self.assertEqual(spec['weight_after_warmup'],.1)
            self.assertEqual(spec['effective_weight_at_save'],expected)
            self.assertTrue(torch.equal(saved['shared_cnl_memory']['valid'],bank.valid))

    def test_historical_configs_still_activate_from_epoch1(self):
        from config import cfg as base
        from .model.make_model_mldc import add_mldc_config
        for name in ['night600_shared_cnl.yml','night600_shared_cnl_consensus.yml']:
            cfg = add_mldc_config(base.clone())
            cfg.merge_from_file(str(Path(__file__).parent/'configs'/name))
            self.assertEqual(cfg.SOLVER.CNL_WARMUP_EPOCHS,0)
            self.assertEqual(auxiliary_weight_for_epoch(cfg,1),.1)
        legacy = SimpleNamespace(SOLVER=SimpleNamespace(AUXILIARY_LOSS='shared_cnl', CNL_WEIGHT=.1))
        self.assertEqual(auxiliary_weight_for_epoch(legacy,1),.1)
        preservation = SimpleNamespace(SOLVER=SimpleNamespace(AUXILIARY_LOSS='preservation',
            PRESERVE_WEIGHT=.1,CNL_WARMUP_EPOCHS=30))
        self.assertEqual(auxiliary_weight_for_epoch(preservation,1),.1)

    def test_invalid_schedule_rejected_before_training(self):
        for warmup in [-1,30.,True]:
            cfg = SimpleNamespace(SOLVER=SimpleNamespace(AUXILIARY_LOSS='shared_cnl',CNL_WEIGHT=.1,CNL_WARMUP_EPOCHS=warmup))
            with self.assertRaises(ValueError):auxiliary_weight_for_epoch(cfg,1)
        cfg = config()
        with self.assertRaises(ValueError):auxiliary_weight_for_epoch(cfg,-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-model',action='store_true')
    parser.add_argument('--report',type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(WarmupChecks))
    if not result.wasSuccessful():raise SystemExit(1)
    report = dict(all_passed=True,unit_checks=result.testsRun,optimizer_steps=0,training_started=False,
        cnl_warmup_epochs=30,effective_weight_epoch30=0.,effective_weight_epoch31=.1)
    if args.full_model:
        from .verify_shared_cnl import full_model_check
        report.update(full_model_check('night600_shared_cnl_consensus_warmup30.yml'))
    if args.report:args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':main()
