import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description='Night600 MLDC checkpoint evaluation')
    parser.add_argument('--config_file', default='mldc_cls_readout/variants/all_cls_random_initialization/config.yml')
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    from config import cfg as base_cfg
    from datasets import make_dataloader
    from utils.logger import setup_logger
    from mldc_cls_readout.model.make_model_mldc import add_mldc_config, make_model
    from mldc_cls_readout.processor.processor_mldc import do_inference
    cfg = add_mldc_config(base_cfg.clone())
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if not cfg.TEST.WEIGHT:
        parser.error('Specify TEST.WEIGHT for this experiment')
    cfg.MODEL.PRETRAIN_CHOICE = 'self'
    cfg.MODEL.PRETRAIN_PATH = cfg.TEST.WEIGHT
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    logger = setup_logger('transreid', cfg.OUTPUT_DIR, if_train=False)
    _, _, loader, nq, nc, cameras, views = make_dataloader(cfg)
    model = make_model(cfg, nc, cameras, views)
    logger.info('Evaluating %s', cfg.TEST.WEIGHT)
    do_inference(cfg, model, loader, nq)


if __name__ == '__main__':
    main()
