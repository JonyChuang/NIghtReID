"""Evaluate an explicitly selected server checkpoint without changing training."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--weight',required=True)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=parser.parse_args()
    import torch
    from config import cfg as base_cfg
    from datasets import make_dataloader
    from mldc_feature_enhancement.model import VARIANT,LEGACY_VARIANT,config_from_dump,FeatEnHancerMLDC
    from mldc_feature_enhancement.processor import evaluate
    saved=torch.load(args.weight,map_location='cpu',weights_only=False,mmap=True)
    if saved['variant'] not in (VARIANT,LEGACY_VARIANT):
        raise ValueError('Not a FeatEnHancer + MLDC fixed-equal checkpoint')
    cfg=config_from_dump(base_cfg,saved['config'])
    cfg.MODEL.DEVICE=args.device
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES']=str(cfg.MODEL.DEVICE_ID)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('No visible CUDA GPU; no implicit CPU fallback')
    _,_,loader,nq,nc,cameras,views=make_dataloader(cfg)
    model=FeatEnHancerMLDC(cfg,nc,cameras,views,initialize=False)
    if model.architecture()!=saved['architecture']:
        raise ValueError('Checkpoint architecture mismatch')
    model.load_state_dict(saved['state_dict'],strict=True)
    model.to(args.device)
    metrics,diagnostics=evaluate(cfg,model,loader,nq,args.device)
    report=dict(checkpoint=args.weight,epoch=saved['epoch'],metrics=metrics,**diagnostics)
    path=Path(args.weight).parent/f"evaluation_epoch{saved['epoch']}.json"
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
