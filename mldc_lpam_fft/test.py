"""Evaluate an explicitly supplied V4 checkpoint on the server."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--weight',required=True)
    parser.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    args=parser.parse_args()
    import torch
    from config import cfg as base_cfg
    from datasets import make_dataloader
    from mldc_lpam_fft.model import VARIANT,add_config,LPAMFFTReID
    from mldc_lpam_fft.processor import evaluate
    saved=torch.load(args.weight,map_location='cpu',weights_only=False,mmap=True)
    if saved['variant']!=VARIANT:
        raise ValueError('Not a V4 FFT checkpoint')
    cfg=add_config(base_cfg.clone());cfg.merge_from_other_cfg(type(cfg).load_cfg(saved['config']))
    cfg.MODEL.DEVICE=args.device
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES']=str(cfg.MODEL.DEVICE_ID)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('No visible CUDA GPU; no CPU fallback')
    _,_,loader,nq,nc,cameras,views=make_dataloader(cfg)
    model=LPAMFFTReID(cfg,nc,cameras,views,initialize=False)
    if model.architecture()!=saved['architecture']:
        raise ValueError('Checkpoint architecture differs')
    model.load_state_dict(saved['state_dict'],strict=True)
    model.to(args.device)
    results,diagnostics=evaluate(cfg,model,loader,nq,args.device)
    report=dict(epoch=saved['epoch'],checkpoint=args.weight,readouts=results,**diagnostics)
    target=Path(args.weight).parent/f"evaluation_epoch{saved['epoch']}.json"
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
