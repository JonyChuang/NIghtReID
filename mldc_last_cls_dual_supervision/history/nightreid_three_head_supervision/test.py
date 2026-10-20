"""Evaluate fusion/last/concat from one strict-loaded checkpoint, no fitting."""
import argparse,os,sys
from datetime import datetime,timezone,timedelta
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--neck-feat',choices=['before','after'],default=None)
    parser.add_argument('--output',default='');args=parser.parse_args()
    import torch
    from config import cfg as base_cfg
    from datasets import make_dataloader
    from mldc_last_cls_dual_supervision.model import VARIANT,DualMLDCReID,config_from_dump
    from mldc_last_cls_dual_supervision.evaluate import evaluate
    saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if saved.get('variant')!=VARIANT:raise ValueError('Expected an independent dual-supervision checkpoint')
    cfg=config_from_dump(base_cfg,saved['config']);cfg.DATALOADER.NUM_WORKERS=0
    if args.neck_feat is not None:cfg.TEST.NECK_FEAT=args.neck_feat
    cfg.TEST.IMS_PER_BATCH=min(int(cfg.TEST.IMS_PER_BATCH),64 if args.device=='cuda' else 4);cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES']=str(cfg.MODEL.DEVICE_ID)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; request --device cpu explicitly for CPU evaluation')
    torch.set_num_threads(1)
    _,_,loader,nq,nc,cameras,views=make_dataloader(cfg)
    model=DualMLDCReID(cfg,nc,cameras,views,initialize=False)
    model.load_state_dict(saved['state_dict'],strict=True,assign=True)
    device=torch.device(args.device);model.to(device).eval()
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output=Path(args.output) if args.output else Path(args.checkpoint).parent/'reevaluation'/(stamp+'_'+cfg.TEST.NECK_FEAT)
    output.mkdir(parents=True,exist_ok=False)
    metrics,comparisons=evaluate(cfg,model,loader,nq,device,output)
    print('EVALUATION_METRICS',metrics,flush=True);print('PAIRED_COMPARISONS',comparisons,flush=True)


if __name__=='__main__':main()
