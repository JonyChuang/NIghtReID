"""Train original MLDC+CNL with a training-only, shared-tail multi-image path."""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_file', default=str(
        Path(__file__).parent / 'configs/mvi_group_nightreid.yml'))
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import numpy as np
    import torch
    from config import cfg as defaults
    from datasets import make_dataloader
    from solver import make_optimizer
    from utils.logger import setup_logger
    from mldc_convstream_texture_cnl.indexed_data import indexed_loader
    from mldc_convstream_texture_cnl.model.make_model_mldc import (
        add_mldc_config, make_model, configure_head_learning_rates)
    from mldc_convstream_texture_cnl.loss.make_loss_mldc import make_loss
    from mldc_convstream_texture_cnl.scheduler_mldc import (
        create_fixed_equal_scheduler, wrap_weight_scheduler)
    from mldc_convstream_texture_cnl.processor.processor_mldc import do_train

    cfg = add_mldc_config(defaults.clone())
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if (not cfg.MVI.ENABLED or cfg.NFC.ENABLED or cfg.NFC_CENTER.ENABLED
            or cfg.PAD.ENABLED or cfg.PEM.ENABLED
            or cfg.DATASETS.NAMES != 'nightreid' or cfg.MODEL.PRETRAIN_CHOICE != 'imagenet'
            or cfg.MODEL.SIE_CAMERA or cfg.MODEL.SIE_VIEW
            or cfg.MODEL.MLDC_READOUT != 'mldc' or cfg.MODEL.MLDC_INIT != 'uniform'
            or list(cfg.MODEL.MLDC_LAYERS) != [9, 10, 11, 12]
            or cfg.TEST.NECK_FEAT != 'after'
            or cfg.SOLVER.CNL_THRESHOLD_REFERENCE != 'all_nonself'
            or cfg.SOLVER.CNL_WEIGHT != .1 or cfg.SOLVER.CNL_GAMMA != .9
            or cfg.SOLVER.SEED != 1234 or cfg.SOLVER.MAX_EPOCHS != 120
            or cfg.SOLVER.WARMUP_EPOCHS != 20 or cfg.SOLVER.WARMUP_METHOD != 'cosine'
            or cfg.MODEL.ID_LOSS_WEIGHT != 1. or cfg.MODEL.TRIPLET_LOSS_WEIGHT != 1.):
        raise ValueError('Require original SSL120 NightReID MLDC+CNL with original ID/Triplet weights')
    if (cfg.MVI.SPLIT_LAYER != 8 or cfg.MVI.GROUP_SIZE != 2
            or not 0 <= cfg.MVI.WARMUP_EPOCHS < cfg.SOLVER.MAX_EPOCHS
            or cfg.MVI.RAMP_EPOCHS < 1 or cfg.MVI.GROUP_ID_WEIGHT < 0
            or cfg.MVI.IP_WEIGHT < 0):
        raise ValueError('Invalid shared-tail grouping or activation settings')
    if not Path(cfg.MODEL.PRETRAIN_PATH).is_file():
        raise FileNotFoundError(cfg.MODEL.PRETRAIN_PATH)
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    if cfg.MODEL.DEVICE != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('A visible CUDA GPU is required')
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
    if (len(dataset), nq, len(val.dataset) - nq, nc) != (15514, 1394, 26111, 500):
        raise ValueError('Unexpected official NightReID split')
    model = make_model(cfg, nc, cameras, views, initialize=True)
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('transreid', str(output), if_train=True)
    logger.info('Config:\n%s', cfg)
    logger.info('MultiView: one backbone; shared blocks9-12, MLDC, BN affine and classifier. '
                'Only original images update BN running statistics and CNL memory. '
                'Original ID/Triplet weights remain 1; one-way normalized-feature IP.')
    metadata = dict(
        variant=model.architecture()['variant'], architecture=model.architecture(),
        config=cfg.dump(), initialization=cfg.MODEL.PRETRAIN_PATH,
        student_init='ssl', source_epoch=0, epochs=cfg.SOLVER.MAX_EPOCHS,
        training_dataset_fingerprint=dataset.fingerprint,
        loss='original ID + original Triplet + 0.1 CNL + ramp*(alpha*group ID + lambda*one-way IP)',
        single_id_coefficient=1., single_triplet_coefficient=1.,
        group_source='current batch; distinct same-ID image indices; cross-camera by default',
        aggregation='post-B8 mean CLS plus concatenated patches through existing blocks9-12',
        bn='single path updates state; group uses detached single batch moments and shared affine',
        ip_target='current group final L2 descriptor, detached only for IP',
        teacher=False, offline_targets=False, new_parameters=0,
        cnl_memory='original single-image descriptors only; no group entries',
        inference='original independent single-image ViT+MLDC+BN', resume_supported=False,
        checkpoint='model and CNL memory; no optimizer resume state',
        method='ViT adaptation inspired by multi-view integration; not an exact MVI2P reproduction')
    (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    loss_fn, center = make_loss(cfg, nc)
    optimizer, _ = make_optimizer(cfg, model, center)
    configure_head_learning_rates(cfg, model, optimizer)
    scheduler = wrap_weight_scheduler(cfg, model, optimizer,
                                     create_fixed_equal_scheduler(cfg, optimizer))
    do_train(cfg, model, loader, val, optimizer, scheduler, loss_fn, nq, dataset)


if __name__ == '__main__':
    main()
