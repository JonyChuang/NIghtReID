"""Run from the TransReID project root; importing this file never trains."""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description='Night600 MLDC + PID-free shared-layer CNL')
    parser.add_argument('--config_file', default=str(Path(__file__).parent / 'configs/night600_shared_cnl.yml'))
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import numpy as np
    import torch
    from config import cfg as base_cfg
    from datasets import make_dataloader
    from solver import make_optimizer, WarmupMultiStepLR
    from solver.scheduler_factory import create_scheduler
    from utils.logger import setup_logger
    from mldc_information_preservation.model.make_model_mldc import add_mldc_config, make_model, configure_head_learning_rates
    from mldc_information_preservation.loss.make_loss_mldc import make_loss
    from mldc_information_preservation.scheduler_mldc import wrap_weight_scheduler
    from mldc_information_preservation.processor.processor_mldc import do_train

    cfg = add_mldc_config(base_cfg.clone())
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    shared_cnl = cfg.SOLVER.AUXILIARY_LOSS == 'shared_cnl'
    if cfg.SOLVER.AUXILIARY_LOSS not in ('preservation', 'shared_cnl'):
        raise ValueError('Auxiliary loss must be preservation or shared_cnl')
    if shared_cnl and (cfg.SOLVER.PRESERVE_WEIGHT != 0 or cfg.SOLVER.CNL_WEIGHT < 0
                       or cfg.SOLVER.PRESERVE_REFERENCE_SPACE != 'shared_bn'):
        raise ValueError('Shared CNL replaces preservation and uses the current fused BN mapping')
    if (cfg.MODEL.MLDC_READOUT != 'mldc' or list(cfg.MODEL.MLDC_LAYERS) != [9, 10, 11, 12]
            or cfg.MODEL.MLDC_INIT != 'uniform' or cfg.MODEL.MLDC_TRAIN_FEATURE != 'mldc'
            or cfg.MODEL.PRETRAIN_CHOICE != 'imagenet' or cfg.TEST.MLDC_RETRIEVAL != 'mldc'
            or cfg.TEST.NECK_FEAT != 'after' or cfg.TEST.FEAT_NORM != 'yes'
            or cfg.SOLVER.PRESERVE_WEIGHT < 0):
        raise ValueError('Use uniform-start learned CLS9-12, original SSL, BN-after retrieval and nonnegative preservation weight')
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
    from mldc_information_preservation.indexed_data import indexed_loader
    train_loader, _, val_loader, nq, nc, cameras, views = make_dataloader(cfg)
    train_loader, dataset = indexed_loader(train_loader)
    metadata = dict(variant=('mldc_information_preservation_bn_ratio_20261021'
                            if cfg.SOLVER.PRESERVE_REFERENCE_SPACE == 'shared_bn' and cfg.SOLVER.PRESERVE_TARGET_MODE == 'ratio'
                            else 'mldc_information_preservation_20261020'), readout='mldc',
        seed=int(cfg.SOLVER.SEED), config=cfg.dump(), initialization=cfg.MODEL.PRETRAIN_PATH,
        baseline_run='/root/log/transreid/night600/vit_base_ics_a3pfn_mldc/deep4_uniform_start_120',
        fusion='learned softmax weights over CLS9-12, uniform initialization',
        temperature=float(cfg.MODEL.MLDC_TEMPERATURE), descriptor_dim=768,
        loss='original ID+Triplet + weighted multilayer information preservation; no cross-camera contrastive CNL',
        preservation=dict(weight=float(cfg.SOLVER.PRESERVE_WEIGHT),
            min_margin=float(cfg.SOLVER.PRESERVE_MIN_MARGIN),
            max_margin=float(cfg.SOLVER.PRESERVE_MAX_MARGIN) if cfg.SOLVER.PRESERVE_TARGET_MODE == 'capped' else None,
            target_mode=cfg.SOLVER.PRESERVE_TARGET_MODE, retain_ratio=float(cfg.SOLVER.PRESERVE_RETAIN_RATIO),
            reference_space=cfg.SOLVER.PRESERVE_REFERENCE_SPACE,
            momentum=float(cfg.SOLVER.PRESERVE_MOMENTUM), hard_negatives=int(cfg.SOLVER.PRESERVE_HARD_NEGATIVES),
            query='L2-normalized BN-after learned fusion',
            reference=('detached CLS9-12 transformed with fused batch BN statistics and affine parameters, then L2'
                       if cfg.SOLVER.PRESERVE_REFERENCE_SPACE == 'shared_bn' else 'detached L2-normalized raw CLS9-12'),
            positive_rule='same identity, different original image; camera unrestricted',
            negative_rule='different identity; union top-K from fusion and each layer',
            reference_rule=('retain ratio of maximum reliable single-layer margin; no upper cap'
                            if cfg.SOLVER.PRESERVE_TARGET_MODE == 'ratio' else 'capped maximum reliable single-layer margin')),
        training_images=len(dataset), training_dataset_fingerprint=dataset.fingerprint,
        primary_endpoint=int(cfg.SOLVER.MAX_EPOCHS), primary_readout='global',
        optimizer_and_schedule='unchanged original baseline solver + configure_head_learning_rates + wrap_weight_scheduler',
        resume_supported=False)
    if shared_cnl:
        metadata.update(variant='mldc_multilayer_mined_retrieval_cnl_20261021',
            loss='original ID+Triplet + weighted PID-free multilayer-mined final-retrieval CNL',
            cnl=dict(weight=float(cfg.SOLVER.CNL_WEIGHT), gamma=float(cfg.SOLVER.CNL_GAMMA),
                temperature=float(cfg.SOLVER.CNL_TEMPERATURE), momentum=float(cfg.SOLVER.CNL_MOMENTUM),
                cross_camera=bool(cfg.SOLVER.CNL_CROSS_CAMERA),
                memory='one image-index memory with [N,4,768] mining keys and [N,768] fused retrieval keys',
                memory_coverage='all training image indices; online fill, only valid historical slots and current images participate',
                matching='same-layer cosine only', positive_rule='union of per-layer relative-threshold neighbors',
                contrast_space='L2-normalized final fused BN-after descriptor',
                aggregation='mean negative log probability over all shared positives, then active queries',
                feature_space='same fused BN statistics and affine mapping, then L2',
                gradient='current fused query to backbone, learnable fusion logits and BN; mining and all keys detached',
                identity_labels_used=False, memory_updates='successful optimizer steps only'))
        del metadata['preservation']
    (Path(cfg.OUTPUT_DIR) / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
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
    do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn, nq, dataset)


if __name__ == '__main__':
    main()
