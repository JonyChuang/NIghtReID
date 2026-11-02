"""Compare every epoch to the actual fixed-equal scheduler; no model training."""
import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import warnings

import torch


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(__file__).resolve().parents[1]
    current = load_module(Path(__file__).with_name('scheduler_mldc.py'), 'learned_cnl_schedule_check')
    reference = load_module(root / 'mldc_feature_enhancement/cppi_optim.py', 'fixed_cnl_schedule_reference')
    cfg = SimpleNamespace(SOLVER=SimpleNamespace(WARMUP_EPOCHS=20, MAX_EPOCHS=120,
                                                MLDC_WEIGHT_LR_FACTOR=1.0))
    names = ['backbone', 'backbone_bias', 'head', 'head_bias', 'fusion_logits']
    base_lrs = [4e-4, 8e-4, 2e-3, 4e-3, 4e-4]

    def fixture():
        parameters = [torch.nn.Parameter(torch.zeros(4)) for _ in names]
        optimizer = torch.optim.SGD([{'params': [p], 'lr': lr}
                                    for p, lr in zip(parameters, base_lrs)], momentum=.9)
        model = SimpleNamespace(mldc=SimpleNamespace(layer_weights=parameters[-1]))
        return optimizer, model

    trajectories = {}
    for factor in [1.0, 2.5]:
        cfg.SOLVER.MLDC_WEIGHT_LR_FACTOR = factor
        optimizer, model = fixture()
        fixed_optimizer, _ = fixture()
        scheduler = current.wrap_weight_scheduler(cfg, model, optimizer,
                      current.create_fixed_equal_scheduler(cfg, optimizer))
        fixed_scheduler = reference.build_scheduler(cfg, fixed_optimizer)
        rows = []
        saved = None
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Detected call of `lr_scheduler.step')
            for epoch in range(121):
                if epoch:
                    scheduler.step()
                    fixed_scheduler.step()
                actual = [g['lr'] for g in optimizer.param_groups]
                expected = [g['lr'] for g in fixed_optimizer.param_groups]
                expected[-1] *= factor
                assert all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)
                           for a, b in zip(actual, expected)), (factor, epoch, actual, expected)
                rows.append(actual)
                if epoch == 60:
                    saved = (copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict()))
            restored_optimizer, restored_model = fixture()
            restored_scheduler = current.wrap_weight_scheduler(cfg, restored_model, restored_optimizer,
                                 current.create_fixed_equal_scheduler(cfg, restored_optimizer))
            restored_optimizer.load_state_dict(saved[0])
            restored_scheduler.load_state_dict(saved[1])
            for epoch in range(61, 121):
                restored_scheduler.step()
                restored = [g['lr'] for g in restored_optimizer.param_groups]
                assert restored == rows[epoch], (factor, epoch, restored, rows[epoch])
        trajectories[str(factor)] = rows

    result = dict(all_passed=True, reference='mldc_feature_enhancement.cppi_optim.build_scheduler',
                  epoch_positions_checked=121, groups_checked=names, fusion_factors_checked=[1., 2.5],
                  scheduler_restore_after_epoch60=True, optimizer_steps=0, training_started=False,
                  schedule={str(e): dict(zip(names, trajectories['1.0'][e]))
                            for e in [0, 1, 10, 20, 60, 100, 119, 120]})
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
