"""Run from the TransReID project root; importing this file never trains."""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description='Night600 frozen-anchor patch-compensated MLDC')
    parser.add_argument('--config_file', default='mldc_patch_grm/variants/frozen_global_patch_grm/config.yml')
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import numpy as np
    import torch
    from config import cfg as base_cfg
    from datasets import make_dataloader
    from solver import make_optimizer, WarmupMultiStepLR
    from solver.scheduler_factory import create_scheduler
    from utils.logger import setup_logger
    from mldc_patch_grm.model.make_model_mldc import add_mldc_config, make_model, configure_head_learning_rates
    from mldc_patch_grm.loss.make_loss_mldc import make_loss
    from mldc_patch_grm.scheduler_mldc import wrap_weight_scheduler
    from mldc_patch_grm.processor.processor_mldc import do_train

    cfg = add_mldc_config(base_cfg.clone())
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    if cfg.MODEL.DEVICE != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('MLDC training requires a visible CUDA GPU')
    seed = cfg.SOLVER.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    output = Path(cfg.OUTPUT_DIR)
    if output.exists() and (any(output.glob('transformer_*.pth')) or
                            (output / 'run_metadata.json').exists()):
        raise FileExistsError('Use a fresh OUTPUT_DIR for each experiment run')
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('transreid', cfg.OUTPUT_DIR, if_train=True)
    logger.info('Running with config:\n%s', cfg)
    (Path(cfg.OUTPUT_DIR) / 'run_metadata.json').write_text(json.dumps({
        'experiment': 'mldc_patch_compensation_20261016',
        'stage': 'frozen_anchor', 'baseline_checkpoint': cfg.MODEL.PATCH_BASELINE_CHECKPOINT,
        'readout': cfg.MODEL.MLDC_READOUT, 'seed': int(cfg.SOLVER.SEED),
        'config': cfg.dump(), 'source_package': 'mldc_patch_grm',
    }, indent=2), encoding='utf-8')
    train_loader, _, val_loader, nq, nc, cameras, views = make_dataloader(cfg)
    model = make_model(cfg, nc, cameras, views)
    loss_fn, center = make_loss(cfg, nc)
    optimizer, _ = make_optimizer(cfg, model, center)
    configure_head_learning_rates(cfg, model, optimizer)
    if cfg.SOLVER.WARMUP_METHOD == 'cosine':
        scheduler = create_scheduler(cfg, optimizer)
    else:
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                     cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_EPOCHS,
                                     cfg.SOLVER.WARMUP_METHOD)
    scheduler = wrap_weight_scheduler(cfg, model, optimizer, scheduler)
    logger.info('MLDC initialization=%s, fusion-weight LR factor=%.1f, initial fusion-weight LR=%.8g',
                cfg.MODEL.MLDC_INIT, cfg.SOLVER.MLDC_WEIGHT_LR_FACTOR,
                next(group['lr'] for group in optimizer.param_groups
                     if any(p is model.mldc.layer_weights for p in group['params']))
                if model.mldc.layer_weights.requires_grad else 0.)
    do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn, nq)


if __name__ == '__main__':
    main()
