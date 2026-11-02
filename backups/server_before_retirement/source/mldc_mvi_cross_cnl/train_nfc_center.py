"""Train MLDC+CNL with reciprocal-neighbor mean shift (or legacy center contrast)."""
import argparse,json,os,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_file',default=str(Path(__file__).parent/'configs/nfc_center_nightreid.yml'))
    parser.add_argument('opts',nargs=argparse.REMAINDER);args=parser.parse_args()
    import numpy as np
    import torch
    from config import cfg as defaults
    from datasets import make_dataloader
    from solver import make_optimizer
    from utils.logger import setup_logger
    from mldc_mvi_cross_cnl.indexed_data import indexed_loader
    from mldc_mvi_cross_cnl.model.make_model_mldc import add_mldc_config,make_model,configure_head_learning_rates
    from mldc_mvi_cross_cnl.loss.make_loss_mldc import make_loss
    from mldc_mvi_cross_cnl.scheduler_mldc import create_fixed_equal_scheduler,wrap_weight_scheduler
    from mldc_mvi_cross_cnl.processor.processor_mldc import do_train
    from mldc_mvi_cross_cnl.nfc_center import BatchNFCCenterLoss,BatchNFCMeanShiftLoss
    cfg=add_mldc_config(defaults.clone());cfg.merge_from_file(args.config_file);cfg.merge_from_list(args.opts)
    if (not cfg.NFC_CENTER.ENABLED or cfg.NFC.ENABLED or cfg.PAD.ENABLED or cfg.PEM.ENABLED
            or cfg.DATASETS.NAMES!='nightreid' or cfg.MODEL.PRETRAIN_CHOICE!='imagenet'
            or cfg.MODEL.SIE_CAMERA or cfg.MODEL.SIE_VIEW or cfg.MODEL.MLDC_READOUT!='mldc'
            or list(cfg.MODEL.MLDC_LAYERS)!=[9,10,11,12] or cfg.MODEL.MLDC_INIT!='uniform'
            or cfg.TEST.NECK_FEAT!='after' or cfg.SOLVER.CNL_THRESHOLD_REFERENCE!='all_nonself'
            or cfg.SOLVER.CNL_WEIGHT!=.1 or cfg.SOLVER.CNL_GAMMA!=.9
            or cfg.SOLVER.SEED!=1234 or cfg.SOLVER.MAX_EPOCHS!=120
            or cfg.SOLVER.WARMUP_EPOCHS!=20 or cfg.SOLVER.WARMUP_METHOD!='cosine'):
        raise ValueError('Require original SSL120-epoch NightReID MLDC+CNL; KL/PAD/PEM disabled')
    if not Path(cfg.MODEL.PRETRAIN_PATH).is_file():raise FileNotFoundError(cfg.MODEL.PRETRAIN_PATH)
    # Validate the additional loss settings before creating output files.
    if cfg.NFC_CENTER.LOSS=='mean_shift':
        center_loss=BatchNFCMeanShiftLoss(cfg.NFC_CENTER.WEIGHT,
            cfg.NFC_CENTER.WARMUP_EPOCHS,cfg.NFC_CENTER.RAMP_EPOCHS)
    else:
        center_loss=BatchNFCCenterLoss(cfg.NFC_CENTER.TEMPERATURE,cfg.NFC_CENTER.WEIGHT,
            cfg.NFC_CENTER.WARMUP_EPOCHS,cfg.NFC_CENTER.RAMP_EPOCHS)
    if cfg.NFC_CENTER.WARMUP_EPOCHS>=cfg.SOLVER.MAX_EPOCHS:
        raise ValueError('NFC center activation must occur before training ends')
    cfg.freeze();os.environ['CUDA_VISIBLE_DEVICES']=cfg.MODEL.DEVICE_ID
    if cfg.MODEL.DEVICE!='cuda' or not torch.cuda.is_available():raise RuntimeError('A visible CUDA GPU is required')
    random.seed(cfg.SOLVER.SEED);np.random.seed(cfg.SOLVER.SEED)
    torch.manual_seed(cfg.SOLVER.SEED);torch.cuda.manual_seed_all(cfg.SOLVER.SEED)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    output=Path(cfg.OUTPUT_DIR)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Choose an empty OUTPUT_DIR')
    loader,_,val,nq,nc,cameras,views=make_dataloader(cfg);loader,dataset=indexed_loader(loader)
    if (len(dataset),nq,len(val.dataset)-nq,nc)!=(15514,1394,26111,500):raise ValueError('Unexpected official NightReID split')
    model=make_model(cfg,nc,cameras,views,initialize=True)
    output.mkdir(parents=True,exist_ok=True)
    logger=setup_logger('transreid',str(output),if_train=True);logger.info('Config:\n%s',cfg)
    metadata=dict(variant=model.architecture()['variant'],architecture=model.architecture(),config=cfg.dump(),
        initialization=cfg.MODEL.PRETRAIN_PATH,student_init='ssl',source_epoch=0,epochs=cfg.SOLVER.MAX_EPOCHS,
        training_dataset_fingerprint=dataset.fingerprint,
        loss='ID+Triplet+0.1 CNL+lambda(epoch)*batch NFC center contrast',
        graph='current detached batch features, duplicate image indices coalesced, unlabeled mutual top2',
        positive='same-ID cross-camera owner; center excludes all copies of anchor image and needs >=2 independent images',
        negative='different-ID current-batch original feature keys, detached',
        identity_filter_on_graph=False,camera_filter_on_graph=False,
        label_usage='CNL-style positive/negative supervision and diagnostic purity; no graph filtering',
        teacher=False,offline_targets=False,cnl_memory_used_for_NFC=False,resume_supported=False,
        trainable='original ViT+MLDC+BN+ID classifier; no additional trainable parameters',
        checkpoint='model/CNL states; no optimizer resume state',
        inference='original single-image student, no NFC, no centers, no teacher or extra heads')
    if cfg.NFC_CENTER.LOSS=='mean_shift':
        metadata.update(loss='ID+Triplet+0.1 CNL+lambda(epoch)*mean neighbor half-squared distance',
            additional_objective='batch reciprocal-neighbor mean shift; replaces center contrast',
            target='own independent reciprocal neighbors; arithmetic mean NOT renormalized',
            positive='no labeled positive-owner selection; one independent neighbor gives pair alignment',
            negative='none in added term; original ID/Triplet/CNL retained',
            label_usage='duplicate metadata validation and diagnostic purity only for the new term',
            minimum_neighbors=1,extra_softmax=False,extra_temperature=False,
            averaging='sum feature dimensions, average neighbors per active anchor, then active anchors')
    (output/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    loss_fn,center=make_loss(cfg,nc);optimizer,_=make_optimizer(cfg,model,center)
    configure_head_learning_rates(cfg,model,optimizer)
    scheduler=wrap_weight_scheduler(cfg,model,optimizer,create_fixed_equal_scheduler(cfg,optimizer))
    do_train(cfg,model,loader,val,optimizer,scheduler,loss_fn,nq,dataset,nfc_center=center_loss)


if __name__=='__main__':main()
