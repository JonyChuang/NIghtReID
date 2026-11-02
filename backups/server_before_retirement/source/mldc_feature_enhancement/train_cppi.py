"""Train the single-descriptor MLDC/CNL revision from original SSL."""
import argparse
import json
import os
import random
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description='Global MLDC + supervised cross-camera CNL, without a shallow branch')
    parser.add_argument('--config_file',default=str(Path(__file__).with_name('variants/global_cls_cross_camera_cnl/config.yml')))
    parser.add_argument('opts',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    from config import cfg as base_cfg
    from mldc_feature_enhancement.cppi_model import add_config,MLDCCNLReID
    cfg=add_config(base_cfg.clone());cfg.merge_from_file(args.config_file);cfg.merge_from_list(args.opts);cfg.freeze()
    if cfg.MODEL.ENHANCEMENT_MODE!='mldc_cnl' or cfg.MODEL.CNN_SD_ENABLED:
        raise ValueError('Use the global-only cppi.yml or cppi_baseline.yml for this revision')
    if cfg.SOLVER.CNL_WEIGHT<0 or cfg.SOLVER.SHALLOW_LOSS_WEIGHT!=0:
        raise ValueError('Use nonnegative CNL weight and zero shallow supervision weight')
    os.environ['CUDA_VISIBLE_DEVICES']=str(cfg.MODEL.DEVICE_ID)
    import numpy as np
    import torch
    if cfg.MODEL.DEVICE!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('Training requires visible CUDA; no CPU fallback')
    if not cfg.MODEL.NO_MARGIN or cfg.MODEL.IF_LABELSMOOTH!='off' or cfg.MODEL.IF_WITH_CENTER!='no':
        raise ValueError('Keep the matched ID + soft-margin Triplet objective')
    output=Path(cfg.OUTPUT_DIR)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output is not empty; select a fresh result directory')
    random.seed(cfg.SOLVER.SEED);np.random.seed(cfg.SOLVER.SEED)
    torch.manual_seed(cfg.SOLVER.SEED);torch.cuda.manual_seed_all(cfg.SOLVER.SEED)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=True
    from datasets import make_dataloader
    from utils.logger import setup_logger
    from mldc_feature_enhancement.cnn_sdcl import indexed_loader
    from mldc_feature_enhancement.cnn_sdcl_processor import do_train
    from mldc_feature_enhancement.cppi_optim import build_optimizer,build_scheduler,lr_summary
    output.mkdir(parents=True,exist_ok=True)
    logger=setup_logger('transreid',str(output),if_train=True)
    logger.info('Global MLDC/CNL configuration:\n%s',cfg.dump())
    loader,_,val_loader,nq,nc,cameras,views=make_dataloader(cfg)
    loader,dataset=indexed_loader(loader)
    model=MLDCCNLReID(cfg,nc,cameras,views)
    assert all(p.requires_grad for block in model.base.blocks for p in block.parameters())
    optimizer=build_optimizer(cfg,model);scheduler=build_scheduler(cfg,optimizer)
    metadata=dict(architecture=model.architecture(),seed=int(cfg.SOLVER.SEED),config=cfg.dump(),
        initialization=cfg.MODEL.PRETRAIN_PATH,training_mode='joint_from_epoch_1',
        loss='global ID+Triplet + weighted single-space supervised cross-camera CNL',
        cnl=dict(weight=float(cfg.SOLVER.CNL_WEIGHT),feature='L2-normalized BN-after fused CLS',
            neighbor_spaces=1,gamma=float(cfg.SOLVER.CNL_GAMMA),temperature=float(cfg.SOLVER.CNL_TEMPERATURE),
            momentum=float(cfg.SOLVER.CNL_MOMENTUM),memory='one training-image bank'),
        training_dataset_fingerprint=dataset.fingerprint,training_images=len(dataset),
        primary_endpoint=int(cfg.SOLVER.MAX_EPOCHS),primary_readout='global',
        startup_lr=lr_summary(optimizer),resume_supported=False)
    (output/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    logger.info('Experiment metadata: %s',metadata)
    do_train(cfg,model,loader,val_loader,optimizer,scheduler,nq,dataset)


if __name__=='__main__':main()
