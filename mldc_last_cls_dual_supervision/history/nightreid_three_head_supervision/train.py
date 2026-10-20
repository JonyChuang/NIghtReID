"""Training is explicit and requires CUDA; import/deployment never launches it."""
import argparse,json,os,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description='Fixed-equal MLDC + last CLS + supervised NightReID-style concat')
    parser.add_argument('--config_file',default=str(Path(__file__).with_name('config.yml')))
    parser.add_argument('opts',nargs=argparse.REMAINDER);args=parser.parse_args()
    from config import cfg as base_cfg
    from mldc_last_cls_dual_supervision.model import add_config,DualMLDCReID
    cfg=add_config(base_cfg.clone());cfg.merge_from_file(args.config_file);cfg.merge_from_list(args.opts);cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES']=str(cfg.MODEL.DEVICE_ID)
    import numpy as np
    import torch
    if cfg.MODEL.DEVICE!='cuda' or not torch.cuda.is_available():raise RuntimeError('CUDA is required for training; no CPU training fallback')
    if not cfg.MODEL.NO_MARGIN or cfg.MODEL.IF_LABELSMOOTH!='off' or cfg.MODEL.IF_WITH_CENTER!='no':
        raise ValueError('Use independent ID and original soft-margin Triplet losses')
    if cfg.MODEL.ID_LOSS_WEIGHT!=1. or cfg.MODEL.TRIPLET_LOSS_WEIGHT!=1.:raise ValueError('Keep ID/Triplet weights at 1; apply the configured three-branch coefficients')
    output=Path(cfg.OUTPUT_DIR)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Output is populated; choose a fresh directory')
    random.seed(cfg.SOLVER.SEED);np.random.seed(cfg.SOLVER.SEED);torch.manual_seed(cfg.SOLVER.SEED);torch.cuda.manual_seed_all(cfg.SOLVER.SEED)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=True
    from datasets import make_dataloader
    from utils.logger import setup_logger
    from mldc_last_cls_dual_supervision.optim import build_optimizer,build_scheduler,lr_summary
    from mldc_last_cls_dual_supervision.processor import do_train
    train_loader,_,val_loader,nq,nc,cameras,views=make_dataloader(cfg)
    model=DualMLDCReID(cfg,nc,cameras,views)
    assert all(p.requires_grad for block in model.base.blocks for p in block.parameters())
    assert all(p.requires_grad for p in model.base.norm.parameters())
    optimizer=build_optimizer(cfg,model);scheduler=build_scheduler(cfg,optimizer)
    output.mkdir(parents=True,exist_ok=True)
    logger=setup_logger('transreid',str(output),if_train=True)
    logger.info('Dual-supervision configuration:\n%s',cfg.dump())
    metadata=dict(variant=model.architecture()['variant'],architecture=model.architecture(),config=cfg.dump(),
        initialization=cfg.MODEL.PRETRAIN_PATH,seed=int(cfg.SOLVER.SEED),num_classes=nc,
        training_mode='joint_from_epoch_1',primary_endpoint=int(cfg.SOLVER.MAX_EPOCHS),primary_branch='concat',
        startup_lr=lr_summary(optimizer),scheduler='unchanged original cosine factory, including shared scalar warmup/floor',
        source='Original SSL checkpoint, not a trained MLDC continuation')
    (output/'run_metadata.json').write_text(json.dumps(metadata,indent=2));logger.info('Experiment metadata: %s',metadata)
    do_train(cfg,model,train_loader,val_loader,optimizer,scheduler,nq)


if __name__=='__main__':main()
