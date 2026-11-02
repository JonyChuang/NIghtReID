"""Learnable MLDC-guided CNN readout, with two ID/Triplet losses and no CNL."""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config_file',
        default=str(
            Path(__file__).resolve().parent
            / 'variants/cnn_fused_cls_guided_learnable_no_cnl/config.yml'
        ),
    )
    parser.add_argument('opts',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    from config import cfg as base_cfg
    from mldc_feature_enhancement.cppi_model import add_config,CNNShallowDeepReID
    cfg=add_config(base_cfg.clone())
    cfg.merge_from_file(args.config_file);cfg.merge_from_list(args.opts);cfg.freeze()
    if (cfg.MODEL.ENHANCEMENT_MODE!='cnn_shallow_deep' or not cfg.MODEL.CNN_SD_ENABLED
            or cfg.MODEL.CNN_SD_QUERY!='fused_cls' or cfg.MODEL.FEATENHANCER_ENABLED
            or cfg.MODEL.CPPI_ENABLED or cfg.SOLVER.CNL_WEIGHT!=0
            or cfg.SOLVER.SHALLOW_LOSS_WEIGHT < 0):
        raise ValueError('Use fused CLS guidance, independent CNN, two equal ID/Triplet losses and no CNL')
    if (cfg.DATALOADER.SAMPLER!='softmax_triplet' or cfg.MODEL.METRIC_LOSS_TYPE!='triplet'
            or not cfg.MODEL.NO_MARGIN or cfg.MODEL.IF_LABELSMOOTH!='off'
            or cfg.MODEL.IF_WITH_CENTER!='no' or cfg.TEST.FEAT_NORM!='yes' or cfg.TEST.RE_RANKING):
        raise ValueError('Keep the original PK sampling, ID/soft-margin Triplet and normalized retrieval')
    os.environ['CUDA_VISIBLE_DEVICES']=str(cfg.MODEL.DEVICE_ID)
    import numpy as np
    import torch
    if cfg.MODEL.DEVICE!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('Training requires visible CUDA')
    output=Path(cfg.OUTPUT_DIR)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output is not empty; select a fresh result directory')
    random.seed(cfg.SOLVER.SEED);np.random.seed(cfg.SOLVER.SEED)
    torch.manual_seed(cfg.SOLVER.SEED);torch.cuda.manual_seed_all(cfg.SOLVER.SEED)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=True
    from datasets import make_dataloader
    from utils.logger import setup_logger
    from mldc_feature_enhancement.cnn_sdcl import indexed_loader
    from mldc_feature_enhancement.cnn_fused_processor import do_train
    from mldc_feature_enhancement.cppi_optim import build_optimizer,build_scheduler,lr_summary
    output.mkdir(parents=True,exist_ok=True)
    logger=setup_logger('transreid',str(output),if_train=True)
    logger.info('Learnable MLDC-guided CNN configuration:\n%s',cfg.dump())
    loader,_,val_loader,nq,nc,cameras,views=make_dataloader(cfg)
    loader,dataset=indexed_loader(loader)
    model=CNNShallowDeepReID(cfg,nc,cameras,views)
    optimizer=build_optimizer(cfg,model);scheduler=build_scheduler(cfg,optimizer)
    metadata=dict(architecture=model.architecture(),seed=int(cfg.SOLVER.SEED),config=cfg.dump(),
        initialization=cfg.MODEL.PRETRAIN_PATH,training_mode='joint_from_epoch_1',
        loss='global ID+Triplet + shallow ID+Triplet; no CNL',cnl_weight=0.,
        cnl_memory=False,training_images=len(dataset),training_dataset_fingerprint=dataset.fingerprint,
        initial_layer_weights=model.readout_weights().detach().cpu().tolist(),
        primary_endpoint=int(cfg.SOLVER.MAX_EPOCHS),primary_readout='concat',
        startup_lr=lr_summary(optimizer),resume_supported=False)
    (output/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    logger.info('Experiment metadata: %s',metadata)
    do_train(cfg,model,loader,val_loader,optimizer,scheduler,nq,dataset)


if __name__=='__main__':
    main()
