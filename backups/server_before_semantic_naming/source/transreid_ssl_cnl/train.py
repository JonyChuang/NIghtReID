"""Run from the original project root. Importing this module never starts training."""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description='Original TransReID-SSL settings + global CNL')
    parser.add_argument('--config_file', default=str(Path(__file__).resolve().parent / 'configs/night600.yml'))
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import numpy as np
    import torch
    from config import cfg as base_cfg
    from datasets import make_dataloader
    from solver import make_optimizer, WarmupMultiStepLR
    from solver.scheduler_factory import create_scheduler
    from utils.logger import setup_logger
    from transreid_ssl_cnl.model import add_cnl_config, make_model
    from transreid_ssl_cnl.loss import make_loss
    from transreid_ssl_cnl.indexed_data import indexed_loader
    from transreid_ssl_cnl.processor import do_train

    cfg = add_cnl_config(base_cfg.clone())
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if cfg.MODEL.PRETRAIN_CHOICE != 'imagenet' or cfg.SOLVER.CNL_WEIGHT < 0:
        raise ValueError('Start from the original SSL initialization and use nonnegative CNL weight')
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    if cfg.MODEL.DEVICE != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Training requires a visible CUDA GPU')
    seed = cfg.SOLVER.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    output = Path(cfg.OUTPUT_DIR)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty output directory for this independent experiment')
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('transreid', cfg.OUTPUT_DIR, if_train=True)
    logger.info('Running with config:\n%s', cfg)
    train_loader, _, val_loader, nq, nc, cameras, views = make_dataloader(cfg)
    train_loader, dataset = indexed_loader(train_loader)
    model = make_model(cfg, nc, cameras, views)
    loss_fn, center = make_loss(cfg, nc)
    # Original solver: no MLDC head/logit multipliers or replacement schedule.
    optimizer, _ = make_optimizer(cfg, model, center)
    if cfg.SOLVER.WARMUP_METHOD == 'cosine':
        scheduler = create_scheduler(cfg, optimizer)
    else:
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                     cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_EPOCHS,
                                     cfg.SOLVER.WARMUP_METHOD)
    metadata = dict(variant='transreid_ssl_cnl_20261022', config=cfg.dump(),
        architecture='original TransReID-SSL final-LayerNorm CLS + original BNNeck/classifier',
        seed=int(seed), descriptor_dim=model.in_planes, retrieval=cfg.TEST.NECK_FEAT,
        loss='original ID+Triplet + weighted supervised global cross-camera CNL',
        cnl=dict(weight=float(cfg.SOLVER.CNL_WEIGHT), gamma=float(cfg.SOLVER.CNL_GAMMA),
            temperature=float(cfg.SOLVER.CNL_TEMPERATURE), momentum=float(cfg.SOLVER.CNL_MOMENTUM),
            feature='L2-normalized actual retrieval descriptor', neighbor_spaces=1),
        training_images=len(dataset), training_dataset_fingerprint=dataset.fingerprint,
        primary_endpoint=int(cfg.SOLVER.MAX_EPOCHS), resume_supported=False,
        optimizer_and_schedule='original make_optimizer and create_scheduler, step(epoch) after epoch')
    (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn, nq, dataset)


if __name__ == '__main__':
    main()
