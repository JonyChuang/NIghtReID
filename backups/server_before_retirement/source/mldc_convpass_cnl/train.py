"""Run from the TransReID project root; importing this file never trains."""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description='NightReID Convpass + learned MLDC/CNL, seed1234')
    parser.add_argument('--config_file', default=str(Path(__file__).resolve().parent/'configs/convpass_first4_frozen_nightreid.yml'))
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import numpy as np
    import torch
    from config import cfg as base_cfg
    from datasets import make_dataloader
    from solver import make_optimizer, WarmupMultiStepLR
    from utils.logger import setup_logger
    from mldc_convpass_cnl.model.make_model_mldc import add_mldc_config, make_model, configure_head_learning_rates
    from mldc_convpass_cnl.loss.make_loss_mldc import make_loss
    from mldc_convpass_cnl.scheduler_mldc import create_fixed_equal_scheduler, wrap_weight_scheduler
    from mldc_convpass_cnl.processor.processor_mldc import do_train

    cfg = add_mldc_config(base_cfg.clone())
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if cfg.SOLVER.SEED!=1234 or cfg.DATASETS.NAMES!='nightreid' or cfg.SOLVER.WARMUP_METHOD!='cosine':
        raise ValueError('Use NightReID, seed1234 and the matched cosine schedule')
    if (cfg.MODEL.MLDC_READOUT != 'mldc' or list(cfg.MODEL.MLDC_LAYERS) != [9, 10, 11, 12]
            or cfg.MODEL.MLDC_INIT != 'uniform' or cfg.MODEL.MLDC_TRAIN_FEATURE != 'mldc'
            or cfg.MODEL.PRETRAIN_CHOICE != 'imagenet' or cfg.TEST.MLDC_RETRIEVAL != 'mldc'
            or cfg.TEST.NECK_FEAT != 'after' or cfg.TEST.FEAT_NORM != 'yes'
            or cfg.SOLVER.CNL_WEIGHT < 0):
        raise ValueError('Use uniform-start learned CLS9-12, original SSL, BN-after retrieval and nonnegative CNL')
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
    if Path(cfg.OUTPUT_DIR).exists() and any(Path(cfg.OUTPUT_DIR).iterdir()):
        raise FileExistsError('Choose an empty independent output directory')
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    logger = setup_logger('transreid', cfg.OUTPUT_DIR, if_train=True)
    logger.info('Running with config:\n%s', cfg)
    from mldc_convpass_cnl.indexed_data import indexed_loader
    train_loader, _, val_loader, nq, nc, cameras, views = make_dataloader(cfg)
    train_loader, dataset = indexed_loader(train_loader)
    model = make_model(cfg, nc, cameras, views)
    metadata = dict(variant='convpass_mldc_cnl_20261026', readout='mldc',
        architecture=model.architecture(), convpass_initialization=model.initialization_report,
        seed=int(cfg.SOLVER.SEED), config=cfg.dump(), initialization=cfg.MODEL.PRETRAIN_PATH,
        baseline_run='/root/log/transreid/nightreid/mldc_learnable_global_cnl/deep_four_uniform_cross_camera_cnl_fixed_equal_schedule',
        fusion='learned softmax weights over CLS9-12, uniform initialization',
        temperature=float(cfg.MODEL.MLDC_TEMPERATURE), descriptor_dim=768,
        loss='original ID+Triplet + weighted single-space supervised cross-camera CNL',
        cnl=dict(weight=float(cfg.SOLVER.CNL_WEIGHT), gamma=float(cfg.SOLVER.CNL_GAMMA),
            temperature=float(cfg.SOLVER.CNL_TEMPERATURE), momentum=float(cfg.SOLVER.CNL_MOMENTUM),
            feature='L2-normalized BN-after actual learned fusion', neighbor_spaces=1),
        training_images=len(dataset), training_dataset_fingerprint=dataset.fingerprint,
        primary_endpoint=int(cfg.SOLVER.MAX_EPOCHS), primary_readout='global',
        optimizer_and_schedule='original SGD groups + head LR factor + fixed-equal group-scaled cosine + fusion LR factor',
        lr_schedule='fixed_equal_group_scaled_cosine' if cfg.SOLVER.WARMUP_METHOD == 'cosine' else cfg.SOLVER.WARMUP_METHOD,
        resume_supported=False)
    (Path(cfg.OUTPUT_DIR) / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    loss_fn, center = make_loss(cfg, nc)
    optimizer, _ = make_optimizer(cfg, model, center)
    configure_head_learning_rates(cfg, model, optimizer)
    if cfg.SOLVER.WARMUP_METHOD == 'cosine':
        scheduler = create_fixed_equal_scheduler(cfg, optimizer)
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
    do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn, nq, dataset)


if __name__ == '__main__':
    main()
