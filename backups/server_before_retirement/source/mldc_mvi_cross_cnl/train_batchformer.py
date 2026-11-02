"""Train NightReID MLDC+CNL with optional CVPR2022 BatchFormer."""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(default_config='batchformer_nightreid.yml'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_file', default=str(Path(__file__).parent/'configs'/default_config))
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import numpy as np
    import torch
    from config import cfg as defaults
    from datasets import make_dataloader
    from solver import make_optimizer
    from utils.logger import setup_logger
    from mldc_mvi_cross_cnl.indexed_data import indexed_loader
    from mldc_mvi_cross_cnl.model.make_model_mldc import (
        add_mldc_config, make_model, configure_head_learning_rates)
    from mldc_mvi_cross_cnl.loss.make_loss_mldc import make_loss
    from mldc_mvi_cross_cnl.scheduler_mldc import (
        create_fixed_equal_scheduler, wrap_weight_scheduler)
    from mldc_mvi_cross_cnl.processor.processor_mldc import do_train

    cfg = add_mldc_config(defaults.clone())
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if cfg.NFC.ENABLED or cfg.NFC_CENTER.ENABLED or cfg.PAD.ENABLED or cfg.PEM.ENABLED:
        raise ValueError('Use the plain MLDC+CNL baseline; disable previous auxiliary methods')
    if (cfg.DATASETS.NAMES != 'nightreid' or cfg.MODEL.PRETRAIN_CHOICE != 'imagenet'
            or cfg.MODEL.SIE_CAMERA or cfg.MODEL.SIE_VIEW
            or cfg.MODEL.MLDC_READOUT != 'mldc' or cfg.MODEL.MLDC_INIT != 'uniform'
            or list(cfg.MODEL.MLDC_LAYERS) != [9, 10, 11, 12]
            or cfg.TEST.NECK_FEAT != 'after'
            or cfg.SOLVER.CNL_THRESHOLD_REFERENCE != 'all_nonself'
            or cfg.SOLVER.CNL_WEIGHT != .1 or cfg.SOLVER.CNL_GAMMA != .9
            or cfg.SOLVER.SEED != 1234 or cfg.SOLVER.MAX_EPOCHS != 120
            or cfg.SOLVER.WARMUP_EPOCHS != 20 or cfg.SOLVER.WARMUP_METHOD != 'cosine'
            or cfg.MODEL.IF_LABELSMOOTH != 'off'
            or cfg.MODEL.ID_LOSS_WEIGHT != 1. or cfg.MODEL.TRIPLET_LOSS_WEIGHT != 1.):
        raise ValueError('Require original SSL120 NightReID MLDC+CNL settings')
    if not Path(cfg.MODEL.PRETRAIN_PATH).is_file():
        raise FileNotFoundError(cfg.MODEL.PRETRAIN_PATH)
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    if cfg.MODEL.DEVICE != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Training requires a visible CUDA GPU')
    random.seed(cfg.SOLVER.SEED)
    np.random.seed(cfg.SOLVER.SEED)
    torch.manual_seed(cfg.SOLVER.SEED)
    torch.cuda.manual_seed_all(cfg.SOLVER.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    output = Path(cfg.OUTPUT_DIR)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty OUTPUT_DIR')
    loader, _, val, nq, nc, cameras, views = make_dataloader(cfg)
    loader, dataset = indexed_loader(loader)
    if (len(dataset), nq, len(val.dataset)-nq, nc) != (15514, 1394, 26111, 500):
        raise ValueError('Unexpected official NightReID split')
    model = make_model(cfg, nc, cameras, views, initialize=True)
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('transreid', str(output), if_train=True)
    logger.info('Config:\n%s', cfg)
    bf = model.batchformer_enabled
    objective = ('0.5*(single CE + BatchFormer CE) + original Triplet + 0.1 CNL'
                 if bf else 'original CE + original Triplet + 0.1 CNL')
    logger.info('Objective: %s; independent single-image validation.', objective)
    metadata = dict(
        variant=model.architecture()['variant'], architecture=model.architecture(),
        config=cfg.dump(), initialization=cfg.MODEL.PRETRAIN_PATH,
        student_init='ssl', source_epoch=0, epochs=cfg.SOLVER.MAX_EPOCHS,
        seed=cfg.SOLVER.SEED, training_dataset_fingerprint=dataset.fingerprint,
        loss=objective, batchformer=bf,
        batchformer_activation='epoch 1; original optimizer LR warmup retained' if bf else None,
        batchformer_lr='backbone LR with existing per-bias multiplier' if bf else None,
        cnl_memory='original single-image descriptor rows only',
        bn='one shared BN on concatenated original/augmented features' if bf else 'original BN',
        inference='original independent single-image ViT+MLDC+BN; no batch attention',
        new_parameters=sum(p.numel() for p in model.batchformer.parameters()) if bf else 0,
        resume_supported=False,
        checkpoint=('model and CNL memory; no optimizer resume state' if bf
                    else 'model, CNL memory and optimizer state; this entry has no resume route'),
        reference='https://github.com/zhihou7/BatchFormer',
        method='BatchFormerV1 transferred to MLDC+CNL ReID; BNNeck and metric losses are ReID adaptations')
    (output/'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    loss_fn, center = make_loss(cfg, nc)
    optimizer, _ = make_optimizer(cfg, model, center)
    configure_head_learning_rates(cfg, model, optimizer)
    scheduler = wrap_weight_scheduler(cfg, model, optimizer, create_fixed_equal_scheduler(cfg, optimizer))
    do_train(cfg, model, loader, val, optimizer, scheduler, loss_fn, nq, dataset)


if __name__ == '__main__':
    main()
