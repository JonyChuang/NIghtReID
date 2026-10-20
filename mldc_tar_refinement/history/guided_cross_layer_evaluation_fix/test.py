"""Evaluate a completed guided-readout checkpoint; importing this file does not run it."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description='Evaluate MLDC-guided cross-attention v4 main retrieval descriptor')
    parser.add_argument('--config_file', default='mldc_tar_refinement/variants/tar_cls_residual_three_heads/config.yml')
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    from config import cfg as base_cfg
    from datasets import make_dataloader
    from utils.logger import setup_logger
    from mldc_tar_refinement.model.make_model_lpam import add_mldc_config, make_model
    from mldc_tar_refinement.processor.processor_lpam import do_inference
    cfg = add_mldc_config(base_cfg.clone())
    cfg.merge_from_file(args.config_file); cfg.merge_from_list(args.opts); cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    if not cfg.TEST.WEIGHT:
        raise ValueError('Specify TEST.WEIGHT with a guided cross-attention v4 checkpoint')
    _, _, val_loader, nq, nc, cameras, views = make_dataloader(cfg)
    model = make_model(cfg, nc, cameras, views, initialize=False)
    model.load_param(cfg.TEST.WEIGHT)
    setup_logger('transreid', cfg.OUTPUT_DIR, if_train=False)
    do_inference(cfg, model, val_loader, nq)


if __name__ == '__main__':
    main()
