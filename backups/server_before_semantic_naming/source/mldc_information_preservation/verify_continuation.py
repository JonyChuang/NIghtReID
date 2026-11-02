"""Checkpoint continuation checks on synthetic tensors, without training."""
import argparse
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from .extend_training import ConstantContinuationLR,continuation_config,restore_memory
from .cnl_schedule import auxiliary_weight_for_epoch
from .verify_consensus_cnl import fixture


def source():
    from .verify_cnl_warmup import config
    cfg=config();bank,*_=fixture()
    return dict(config=cfg.dump(),epoch=120,shared_cnl_memory=bank.state_dict(),
        shared_cnl_spec=dict(memory_schema='image_index_fused_and_four_layers_v2'),
        training_dataset_fingerprint='example')


class ContinuationChecks(unittest.TestCase):
    def test_config_continues_121_through_150_without_repeating_warmup(self):
        saved=source();cfg=continuation_config(saved,'old/transformer_120.pth',30,'new')
        self.assertEqual(cfg.SOLVER.MAX_EPOCHS,150)
        self.assertEqual(cfg.MODEL.PRETRAIN_CHOICE,'self')
        self.assertEqual(cfg.MODEL.PRETRAIN_PATH,'old/transformer_120.pth')
        self.assertEqual(cfg.SOLVER.CNL_WARMUP_EPOCHS,30)
        self.assertEqual(auxiliary_weight_for_epoch(cfg,121),.1)
        self.assertEqual(cfg.SOLVER.CNL_MIN_SUPPORT,3)

    def test_memory_restore_keeps_all_keys_and_rejects_changed_indices(self):
        saved=source();cfg=continuation_config(saved,'old',30,'new')
        model=SimpleNamespace(base=SimpleNamespace(in_planes=3),mldc=SimpleNamespace(selected_layers=[9,10,11,12]))
        cameras=saved['shared_cnl_memory']['cameras'].tolist()
        dataset=SimpleNamespace(fingerprint='example',entries=[('x',None,c) for c in cameras])
        memory=restore_memory(cfg,model,dataset,saved,'cpu')
        self.assertTrue(all(torch.equal(v,saved['shared_cnl_memory'][k]) for k,v in memory.state_dict().items()))
        dataset.fingerprint='changed'
        with self.assertRaises(ValueError):restore_memory(cfg,model,dataset,saved,'cpu')
        dataset.fingerprint='example';dataset.entries[0]=('x',None,99)
        with self.assertRaises(ValueError):restore_memory(cfg,model,dataset,saved,'cpu')

    def test_constant_learning_rate_does_not_restart_at_base_or_decay_again(self):
        p=torch.nn.Parameter(torch.ones(1));q=torch.nn.Parameter(torch.ones(1))
        optimizer=torch.optim.SGD([dict(params=[p],lr=.0004),dict(params=[q],lr=.002)],momentum=.9)
        schedule=ConstantContinuationLR(optimizer,8e-7)
        for epoch in range(121,151):
            schedule.step(epoch)
            self.assertEqual([g['lr'] for g in optimizer.param_groups],[8e-7,8e-7])
        self.assertEqual(len(optimizer.state),0)
        schedule.load_state_dict(schedule.state_dict())

    def test_original_cosine_floor_matches_selected_continuation_rate(self):
        from .verify_cnl_warmup import config
        from solver.scheduler_factory import create_scheduler
        p=torch.nn.Parameter(torch.ones(1));q=torch.nn.Parameter(torch.ones(1))
        optimizer=torch.optim.SGD([dict(params=[p],lr=.0004),dict(params=[q],lr=.002)],momentum=.9)
        scheduler=create_scheduler(config(),optimizer)
        scheduler.step(120)
        for g in optimizer.param_groups:self.assertAlmostEqual(g['lr'],8e-7,places=14)

    def test_new_checkpoint_saves_optimizer_scheduler_and_scaler(self):
        from .verify_cnl_warmup import config
        from .processor.processor_mldc import save_checkpoint
        model=torch.nn.Linear(3,3)
        model.mldc=SimpleNamespace(selected_layers=[9,10,11,12],temperature=.9)
        model.readout='mldc'
        optimizer=torch.optim.SGD(model.parameters(),lr=8e-7,momentum=.9)
        scheduler=ConstantContinuationLR(optimizer,8e-7)
        scaler=torch.amp.GradScaler('cpu',enabled=False)
        bank,*_=fixture();stream=io.BytesIO()
        save_checkpoint(model,config(),150,stream,memory=bank,optimizer=optimizer,scheduler=scheduler,scaler=scaler)
        stream.seek(0);saved=torch.load(stream,weights_only=False)
        self.assertIn('optimizer_state',saved)
        self.assertEqual(saved['scheduler_state'],scheduler.state_dict())
        self.assertIn('scaler_state',saved)
        self.assertEqual(saved['epoch'],150)
        self.assertEqual(saved['shared_cnl_spec']['effective_weight_at_save'],.1)

    def test_missing_memory_or_incomplete_run_is_rejected(self):
        saved=source();saved['shared_cnl_memory']=None
        with self.assertRaises(ValueError):continuation_config(saved,'old',30,'new')
        saved=source();saved['epoch']=80
        with self.assertRaises(ValueError):continuation_config(saved,'old',30,'new')
        for extra in [0,-1,1.5]:
            with self.assertRaises(ValueError):continuation_config(source(),'old',extra,'new')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path)
    args=parser.parse_args();torch.set_num_threads(1)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContinuationChecks))
    if not result.wasSuccessful():raise SystemExit(1)
    report=dict(all_passed=True,unit_checks=result.testsRun,first_epoch=121,final_epoch=150,
        learning_rate=8e-7,optimizer_steps=0,training_started=False)
    if args.report:args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':main()
