"""Train only on the authorized server with a visible CUDA GPU."""
import argparse
import json
import os
import random
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description='Author FFT LPAM V4, fixed equal last-four global CLS')
    parser.add_argument('--config_file',default=str(Path(__file__).with_name('variants/fixed_global_lpam_fft_concat/config.yml')))
    parser.add_argument('opts',nargs=argparse.REMAINDER)
    args = parser.parse_args()
    from config import cfg as base_cfg
    from mldc_lpam_fft.model import add_config, LPAMFFTReID, VARIANT
    cfg = add_config(base_cfg.clone())
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.MODEL.DEVICE_ID)
    import numpy as np
    import torch
    if cfg.MODEL.DEVICE != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU: training was not started; no CPU or local fallback')
    if not cfg.MODEL.NO_MARGIN or cfg.MODEL.IF_LABELSMOOTH != 'off' or cfg.MODEL.IF_WITH_CENTER != 'no':
        raise ValueError('Retain baseline unsmoothed ID + soft-margin Triplet')
    output = Path(cfg.OUTPUT_DIR)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output is not empty; use a fresh run, never overwrite an experiment')
    random.seed(cfg.SOLVER.SEED); np.random.seed(cfg.SOLVER.SEED)
    torch.manual_seed(cfg.SOLVER.SEED); torch.cuda.manual_seed_all(cfg.SOLVER.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    from datasets import make_dataloader
    from loss.center_loss import CenterLoss
    from solver import make_optimizer, WarmupMultiStepLR
    from solver.scheduler_factory import create_scheduler
    from utils.logger import setup_logger
    from mldc_lpam_fft.processor import do_train
    output.mkdir(parents=True,exist_ok=False)
    logger = setup_logger('transreid',str(output),if_train=True)
    logger.info('V4 config:\n%s',cfg.dump())
    train_loader, _, val_loader, nq, nc, cameras, views = make_dataloader(cfg)
    model = LPAMFFTReID(cfg,nc,cameras,views)
    assert all(p.requires_grad for block in model.base.blocks for p in block.parameters())
    center = CenterLoss(num_classes=nc,feat_dim=768,use_gpu=False)
    optimizer,_ = make_optimizer(cfg,model,center)
    heads = {id(p) for n,p in model.named_parameters() if n.startswith(('bottleneck.','classifier.')) and p.requires_grad}
    for group in optimizer.param_groups:
        ids = {id(p) for p in group['params']}
        if ids & heads:
            if not ids <= heads:
                raise RuntimeError('Mixed backbone/head optimizer group')
            group['lr'] *= cfg.SOLVER.MLDC_HEAD_LR_FACTOR
    if cfg.SOLVER.WARMUP_METHOD=='cosine':
        scheduler = create_scheduler(cfg,optimizer)
    else:
        scheduler = WarmupMultiStepLR(optimizer,cfg.SOLVER.STEPS,cfg.SOLVER.GAMMA,cfg.SOLVER.WARMUP_FACTOR,cfg.SOLVER.WARMUP_EPOCHS,cfg.SOLVER.WARMUP_METHOD)
    metadata = dict(variant=VARIANT,fft=True,architecture=model.architecture(),seed=int(cfg.SOLVER.SEED),
                    initialization=cfg.MODEL.PRETRAIN_PATH,training_mode='joint_from_epoch_1',
                    config=cfg.dump(),loss='mean(branch ID) + global Triplet',
                    primary_endpoint=cfg.SOLVER.MAX_EPOCHS,
                    note='Fixed equal means weights remain 0.25, not a learnable uniform initialization')
    (output/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    logger.info('Experiment metadata: %s',metadata)
    do_train(cfg,model,train_loader,val_loader,optimizer,scheduler,nq)


if __name__=='__main__':
    main()
