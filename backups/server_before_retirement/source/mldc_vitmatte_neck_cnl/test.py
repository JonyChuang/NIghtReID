"""Evaluate a trained transfer checkpoint with the official full query/gallery."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--weight',required=True)
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    args=parser.parse_args()
    import torch
    from yacs.config import CfgNode
    from datasets import make_dataloader
    from .model.make_model_mldc import MLDCReID
    from .processor.processor_mldc import evaluate
    checkpoint=torch.load(args.weight,map_location='cpu',mmap=True,weights_only=False)
    cfg=CfgNode.load_cfg(checkpoint['config']);cfg.defrost();cfg.MODEL.DEVICE=args.device;cfg.freeze()
    _,_,loader,nq,nc,cameras,views=make_dataloader(cfg)
    model=MLDCReID(cfg,nc,cameras,views,initialize=False)
    if checkpoint['architecture']!=model.architecture():raise ValueError('Incompatible checkpoint')
    model.load_state_dict(checkpoint['state_dict'],strict=True);del checkpoint
    device=torch.device(args.device);model.to(device)
    metrics,diagnostics=evaluate(cfg,model,loader,nq,device)
    report=Path(args.weight).with_suffix('.evaluation.json')
    report.write_text(json.dumps(dict(metrics=metrics,diagnostics=diagnostics),indent=2))
    print(json.dumps(dict(metrics=metrics,readouts=diagnostics['readouts'],report=str(report))))


if __name__=='__main__':main()
