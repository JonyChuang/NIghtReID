"""Evaluate a saved TransReID-SSL+CNL checkpoint with its original retrieval settings."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default=str(Path(__file__).resolve().parent / 'configs/night600.yml'))
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import torch
    from config import cfg as base_cfg
    from datasets import make_dataloader
    from transreid_ssl_cnl.model import add_cnl_config, make_model
    from transreid_ssl_cnl.processor import evaluate

    cfg = add_cnl_config(base_cfg.clone())
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    if not cfg.TEST.WEIGHT:
        raise ValueError('Supply TEST.WEIGHT with the experiment checkpoint path')
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    _, _, val_loader, nq, nc, cameras, views = make_dataloader(cfg)
    inference_cfg = cfg.clone()
    inference_cfg.defrost()
    inference_cfg.MODEL.PRETRAIN_CHOICE = 'self'
    inference_cfg.MODEL.PRETRAIN_PATH = cfg.TEST.WEIGHT
    inference_cfg.freeze()
    model = make_model(inference_cfg, nc, cameras, views).to(torch.device(cfg.MODEL.DEVICE))
    metrics = evaluate(cfg, model, val_loader, nq, torch.device(cfg.MODEL.DEVICE))
    print(json.dumps(dict(epoch=model.checkpoint_epoch, **metrics), indent=2))


if __name__ == '__main__':
    main()
