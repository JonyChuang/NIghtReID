"""Import-safe entrypoint. Training starts only when explicitly invoked."""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description='Night600 MLDC + free-query LPAM; existing experiment, no FFT')
    parser.add_argument('--config_file', default='mldc_tar_refinement/variants/tar_cls_residual_three_heads/config.yml')
    parser.add_argument('--resume', default='', help='Full free-query v2 checkpoint, usually transformer_latest.pth')
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import numpy as np
    import torch
    from config import cfg as base_cfg
    from datasets import make_dataloader
    from solver import WarmupMultiStepLR
    from solver.scheduler_factory import create_scheduler
    from utils.logger import setup_logger
    from mldc_tar_refinement.model.make_model_lpam import add_mldc_config, make_model, VARIANT
    from mldc_tar_refinement.loss.make_loss_lpam import make_loss
    from mldc_tar_refinement.processor.processor_lpam import do_train

    cfg = add_mldc_config(base_cfg.clone())
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    if cfg.MODEL.DEVICE != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Training requires a visible CUDA GPU; no training was started')
    warmup, total = cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.MAX_EPOCHS
    if not (0 <= warmup < total):
        raise ValueError('Require 0 <= warmup < total')
    if cfg.DATALOADER.SAMPLER != 'softmax_triplet' or cfg.DATALOADER.NUM_INSTANCE < 2:
        raise ValueError('Use a PK sampler with at least two images per identity')
    if cfg.SOLVER.OPTIMIZER_NAME != 'SGD' or cfg.MODEL.DIST_TRAIN:
        raise ValueError('This entrypoint uses single-GPU SGD')
    output = Path(cfg.OUTPUT_DIR)
    if not args.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError('Run output already exists. Use --resume with its latest checkpoint, or set OUTPUT_DIR.')
    if args.resume and not (output/'run_metadata.json').exists():
        raise ValueError('Resume must use the original run output directory')
    random.seed(cfg.SOLVER.SEED); np.random.seed(cfg.SOLVER.SEED); torch.manual_seed(cfg.SOLVER.SEED)
    torch.cuda.manual_seed_all(cfg.SOLVER.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    train_loader, _, val_loader, nq, nc, cameras, views = make_dataloader(cfg)
    model = make_model(cfg, nc, cameras, views, initialize=not bool(args.resume))
    resume = model.load_param(args.resume) if args.resume else None
    if resume and (resume.get('config') != cfg.dump() or int(resume['epoch']) >= cfg.SOLVER.MAX_EPOCHS):
        raise ValueError('Resume config differs, or the scheduled experiment has already finished')
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('transreid', str(output), if_train=True)
    logger.info('Running with config:\n%s', cfg)
    if not resume:
        metadata = {'variant': VARIANT, 'fft': False, 'readout': 'mldc', 'seed': int(cfg.SOLVER.SEED),
                    'architecture': model.architecture(), 'descriptor_dim': model.descriptor_dim,
                    'training_mode': 'joint_from_start',
                    'backbone_initialization': cfg.MODEL.PRETRAIN_CHOICE,
                    'backbone_pretrain_path': cfg.MODEL.PRETRAIN_PATH if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet' else None,
                    'local_similarity_weight': model.local_weight,
                    'loss': 'G_ID + G_triplet + LPAM_LOCAL_WEIGHT*(local_whole_ID + local_triplet) + LPAM_DIVERSITY_WEIGHT*map_overlap; all terms from epoch 1',
                    'primary_endpoint': cfg.SOLVER.MAX_EPOCHS,
                    'lr_policy': 'original_mldc_scheduler',
                    'global_head_lr_factor': cfg.SOLVER.MLDC_HEAD_LR_FACTOR,
                    'local_head_lr_factor': cfg.SOLVER.LPAM_HEAD_LR_FACTOR,
                    'config': cfg.dump()}
        (output/'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    optimizer = torch.optim.SGD(model.optimizer_groups(cfg), momentum=cfg.SOLVER.MOMENTUM)
    if cfg.SOLVER.WARMUP_METHOD == 'cosine':
        scheduler = create_scheduler(cfg, optimizer)
    else:
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                     cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_EPOCHS,
                                     cfg.SOLVER.WARMUP_METHOD)
    loss_fn = make_loss(cfg, nc)
    do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn, nq, resume=resume)


if __name__ == '__main__':
    main()
