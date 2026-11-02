"""Evaluate the final-CLS ablation with the same retrieval protocol."""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config_file',default='mldc_cnl_isolation/configs/last_cls_cnl01.yml')
    parser.add_argument('opts',nargs=argparse.REMAINDER);args=parser.parse_args()
    from config import cfg as base_cfg
    from datasets import make_dataloader
    from utils.logger import setup_logger
    from mldc_learnable_global_cnl.model.make_model_mldc import add_mldc_config,make_model
    from mldc_cnl_isolation.processor import do_inference
    cfg=add_mldc_config(base_cfg.clone());cfg.merge_from_file(args.config_file);cfg.merge_from_list(args.opts)
    if not cfg.TEST.WEIGHT:parser.error('Supply TEST.WEIGHT')
    cfg.MODEL.PRETRAIN_CHOICE='self';cfg.MODEL.PRETRAIN_PATH=cfg.TEST.WEIGHT;cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES']=cfg.MODEL.DEVICE_ID
    setup_logger('transreid',cfg.OUTPUT_DIR,if_train=False)
    _,_,loader,nq,nc,cameras,views=make_dataloader(cfg)
    do_inference(cfg,make_model(cfg,nc,cameras,views),loader,nq)


if __name__=='__main__':main()
