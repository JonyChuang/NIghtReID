"""Explicit PFE entry; importing/installing this file never starts a run."""
import argparse
import json
import os
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description='PFE inside ViT plus fixed equal CLS9-12')
    parser.add_argument('--config_file', default=str(Path(__file__).with_name('variants/progressive_patch_enhancement/config.yml')))
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    from config import cfg as base_cfg
    from mldc_feature_enhancement.pfe_model import add_config, PFEReID
    cfg = add_config(base_cfg.clone()); cfg.merge_from_file(args.config_file); cfg.merge_from_list(args.opts); cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.MODEL.DEVICE_ID)
    import numpy as np
    import torch
    if cfg.MODEL.DEVICE != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('PFE training requires visible CUDA; no CPU fallback')
    if not cfg.MODEL.NO_MARGIN or cfg.MODEL.IF_LABELSMOOTH != 'off' or cfg.MODEL.IF_WITH_CENTER != 'no':
        raise ValueError('Keep the same full ID + soft-margin Triplet objective')
    output = Path(cfg.OUTPUT_DIR)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('PFE output is not empty; select a fresh output directory')
    random.seed(cfg.SOLVER.SEED); np.random.seed(cfg.SOLVER.SEED)
    torch.manual_seed(cfg.SOLVER.SEED); torch.cuda.manual_seed_all(cfg.SOLVER.SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = True
    from datasets import make_dataloader
    from loss.center_loss import CenterLoss
    from solver import make_optimizer, WarmupMultiStepLR
    from solver.scheduler_factory import create_scheduler
    from utils.logger import setup_logger
    from mldc_feature_enhancement.pfe_processor import do_train
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('transreid', str(output), if_train=True)
    logger.info('PFE configuration:\n%s', cfg.dump())
    train_loader, _, val_loader, nq, nc, cameras, views = make_dataloader(cfg)
    model = PFEReID(cfg, nc, cameras, views)
    assert all(p.requires_grad for b in model.base.blocks for p in b.parameters())
    assert all(p.requires_grad for p in model.pfe_adapters.parameters())
    center = CenterLoss(num_classes=nc, feat_dim=768, use_gpu=False)
    optimizer, _ = make_optimizer(cfg, model, center)
    heads = {id(p) for n, p in model.named_parameters() if n.startswith(('bottleneck.', 'classifier.')) and p.requires_grad}
    for group in optimizer.param_groups:
        ids = {id(p) for p in group['params']}
        if ids & heads:
            if not ids <= heads:
                raise RuntimeError('Mixed backbone/head optimizer group')
            group['lr'] *= cfg.SOLVER.MLDC_HEAD_LR_FACTOR
    if cfg.SOLVER.WARMUP_METHOD == 'cosine':
        scheduler = create_scheduler(cfg, optimizer)
    else:
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)
    metadata = dict(variant=model.architecture()['variant'], architecture=model.architecture(),
                    seed=int(cfg.SOLVER.SEED), initialization=cfg.MODEL.PRETRAIN_PATH,
                    training_mode='joint_from_epoch_1', config=cfg.dump(), primary_endpoint=cfg.SOLVER.MAX_EPOCHS,
                    loss='global ID + global Triplet, weights 1 and 1',
                    note='Paper-based core adaptation; no A/B branch or detection-specific head/loss')
    (output/'run_metadata.json').write_text(json.dumps(metadata, indent=2))
    logger.info('PFE experiment metadata: %s', metadata)
    do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, nq)


if __name__ == '__main__':
    main()
