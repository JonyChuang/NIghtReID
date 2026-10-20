"""Run from the server project root. Importing this file never trains."""
import argparse
import json
import os
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description='Joint deep-four CLS and BoQ patch readout')
    parser.add_argument('--config_file', default='mldc_boq_local_readout/variants/middle_patch_multi_query_boq/config.yml')
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import numpy as np
    import torch
    from config import cfg as base_cfg
    from datasets import make_dataloader
    from solver import make_optimizer, WarmupMultiStepLR
    from solver.scheduler_factory import create_scheduler
    from loss.center_loss import CenterLoss
    from utils.logger import setup_logger
    from mldc_boq_local_readout.model import add_config, build_model, EXPERIMENT
    from mldc_boq_local_readout.processor import do_train
    cfg = add_config(base_cfg.clone())
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    if cfg.MODEL.DEVICE != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Full joint training requires a visible CUDA GPU; no CPU or local fallback')
    if cfg.SOLVER.MLDC_WEIGHT_LR_FACTOR != 1.0 or cfg.SOLVER.LOCAL_LOSS_WEIGHT <= 0:
        raise ValueError('Use the matched MLDC weight schedule and positive local supervision')
    seed = cfg.SOLVER.SEED
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    output = Path(cfg.OUTPUT_DIR)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh output directory; this entry does not resume experiments')
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('transreid', str(output), if_train=True)
    train_loader, _, val_loader, nq, nc, cameras, views = make_dataloader(cfg)
    model = build_model(cfg, nc, cameras, views)
    # Center loss is unused; retain the baseline optimizer builder's interface.
    center = CenterLoss(num_classes=nc, feat_dim=768, use_gpu=False)
    optimizer, _ = make_optimizer(cfg, model, center)
    head_prefixes = ('global_model.bottleneck.', 'global_model.classifier.', 'local_bn.', 'local_classifier.')
    head_ids = {id(p) for n,p in model.named_parameters() if n.startswith(head_prefixes) and p.requires_grad}
    for group in optimizer.param_groups:
        ids = {id(p) for p in group['params']}
        if ids & head_ids:
            if not ids <= head_ids:
                raise RuntimeError('Optimizer mixed classification heads with other parameters')
            group['lr'] *= cfg.SOLVER.MLDC_HEAD_LR_FACTOR
    if cfg.SOLVER.WARMUP_METHOD == 'cosine':
        scheduler = create_scheduler(cfg, optimizer)
    else:
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                     cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)
    metadata = {'experiment':EXPERIMENT, 'training_mode':'joint_from_epoch_1',
                'architecture':model.architecture(), 'initialization':'same SSL as deep4 baseline; no trained ReID checkpoint',
                'config':cfg.dump(), 'primary_endpoint':cfg.SOLVER.MAX_EPOCHS}
    (output/'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    logger.info('Joint local experiment: %s', metadata)
    do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, nq)


if __name__ == '__main__':
    main()
