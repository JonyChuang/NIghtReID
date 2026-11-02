"""Independent student training entry point; no download or training on import."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default=str(Path(__file__).parent/'configs/convnext_relation_nightreid.yml'))
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import numpy as np
    import torch
    from config import cfg as defaults
    from datasets import make_dataloader
    from solver import make_optimizer
    from utils.logger import setup_logger
    from mldc_convnext_relation_cnl.indexed_data import indexed_loader
    from mldc_convnext_relation_cnl.model.make_model_mldc import add_mldc_config, make_model, configure_head_learning_rates
    from mldc_convnext_relation_cnl.loss.make_loss_mldc import make_loss
    from mldc_convnext_relation_cnl.scheduler_mldc import create_fixed_equal_scheduler, wrap_weight_scheduler
    from mldc_convnext_relation_cnl.processor.processor_mldc import do_train
    cfg = add_mldc_config(defaults.clone())
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if (cfg.DATASETS.NAMES != 'nightreid' or cfg.SOLVER.SEED != 1234 or cfg.SOLVER.MAX_EPOCHS != 120
            or cfg.SOLVER.WARMUP_EPOCHS != 20 or cfg.SOLVER.WARMUP_METHOD != 'cosine'
            or cfg.SOLVER.BASE_LR != .0004 or cfg.SOLVER.IMS_PER_BATCH != 64
            or cfg.SOLVER.OPTIMIZER_NAME != 'SGD'
            or cfg.MODEL.SIE_CAMERA or cfg.MODEL.SIE_VIEW or cfg.MODEL.PRETRAIN_CHOICE != 'imagenet'
            or cfg.MODEL.MLDC_READOUT != 'mldc' or list(cfg.MODEL.MLDC_LAYERS) != [9, 10, 11, 12]
            or cfg.MODEL.MLDC_INIT != 'uniform' or cfg.TEST.NECK_FEAT != 'after'
            or cfg.SOLVER.CNL_WEIGHT != .1 or cfg.SOLVER.CNL_GAMMA != .9):
        raise ValueError('Preserve seed1234, 120 epochs, batch64 and the original MLDC+CNL schedule')
    if cfg.PATCH_KD.ENABLED and not Path(cfg.PATCH_KD.TEACHER_PATH).is_file():
        from mldc_convnext_relation_cnl.model.convnext_teacher import WEIGHTS_URL
        raise FileNotFoundError('Download {} manually to {}'.format(WEIGHTS_URL, cfg.PATCH_KD.TEACHER_PATH))
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    if cfg.MODEL.DEVICE != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Training requires a visible CUDA GPU')
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    loader, _, val, nq, nc, cameras, views = make_dataloader(cfg)
    loader, dataset = indexed_loader(loader)
    assert (len(dataset), nq, len(val.dataset)-nq, nc) == (15514, 1394, 26111, 500)
    model = make_model(cfg, nc, cameras, views)
    output = Path(cfg.OUTPUT_DIR)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty independent output directory')
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('transreid', str(output), if_train=True)
    logger.info('Config:\n%s', cfg)
    metadata = dict(variant='convnext_patch_relation_20261027', architecture=model.architecture(),
        seed=1234, config=cfg.dump(), initialization=cfg.MODEL.PRETRAIN_PATH,
        teacher_metadata=model.texture_teacher.provenance if model.texture_teacher is not None else None,
        training_dataset_fingerprint=dataset.fingerprint, training_images=len(dataset),
        teacher_nightreid_training_epochs=0, primary_endpoint=120, descriptor_dim=768,
        training_started=True, resume_supported=False)
    (output/'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    loss_fn, center = make_loss(cfg, nc)
    optimizer, _ = make_optimizer(cfg, model, center)
    configure_head_learning_rates(cfg, model, optimizer)
    optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
    if model.texture_teacher is not None:
        assert all(id(p) not in optimized for p in model.texture_teacher.parameters())
    scheduler = wrap_weight_scheduler(cfg, model, optimizer, create_fixed_equal_scheduler(cfg, optimizer))
    do_train(cfg, model, loader, val, optimizer, scheduler, loss_fn, nq, dataset)


if __name__ == '__main__':
    main()
