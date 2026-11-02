"""Fine-tune the existing single-image MLDC+CNL with fixed NFC class probabilities."""
import argparse,json,os,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_file',default=str(Path(__file__).parent/'configs/nfc_class_kl_nightreid.yml'))
    parser.add_argument('opts',nargs=argparse.REMAINDER);args=parser.parse_args()
    import numpy as np
    import torch
    from config import cfg as defaults
    from datasets import make_dataloader
    from solver import make_optimizer
    from utils.logger import setup_logger
    from mldc_convstream_texture_cnl.indexed_data import indexed_loader
    from mldc_convstream_texture_cnl.model.make_model_mldc import add_mldc_config,make_model,configure_head_learning_rates
    from mldc_convstream_texture_cnl.loss.make_loss_mldc import make_loss
    from mldc_convstream_texture_cnl.scheduler_mldc import create_fixed_equal_scheduler,wrap_weight_scheduler
    from mldc_convstream_texture_cnl.processor.processor_mldc import do_train
    from mldc_convstream_texture_cnl.nfc_class_distillation import build_fixed_targets
    cfg=add_mldc_config(defaults.clone());cfg.merge_from_file(args.config_file);cfg.merge_from_list(args.opts)
    if (not cfg.NFC.ENABLED or cfg.PAD.ENABLED or cfg.PEM.ENABLED or cfg.DATASETS.NAMES!='nightreid'
            or cfg.MODEL.SIE_CAMERA or cfg.MODEL.SIE_VIEW or cfg.MODEL.MLDC_READOUT!='mldc'
            or list(cfg.MODEL.MLDC_LAYERS)!=[9,10,11,12] or cfg.TEST.NECK_FEAT!='after'
            or cfg.SOLVER.CNL_THRESHOLD_REFERENCE!='all_nonself' or cfg.SOLVER.CNL_GAMMA!=.9
            or cfg.SOLVER.CNL_WEIGHT!=.1 or cfg.NFC.TARGET not in ['raw','nfc']
            or cfg.NFC.TEMPERATURE<=0 or cfg.NFC.WEIGHT<0
            or cfg.SOLVER.MAX_EPOCHS<=cfg.SOLVER.WARMUP_EPOCHS):
        raise ValueError('Require original NightReID MLDC+CNL, PAD/PEM off, valid NFC fine-tuning settings')
    checkpoint=Path(cfg.NFC.TEACHER_CHECKPOINT)
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    if cfg.MODEL.PRETRAIN_PATH!=str(checkpoint):raise ValueError('Student and teacher must initialize from the SAME baseline')
    cfg.freeze();os.environ['CUDA_VISIBLE_DEVICES']=cfg.MODEL.DEVICE_ID
    if not torch.cuda.is_available():raise RuntimeError('Start this training command on a visible CUDA GPU')
    def seed():
        random.seed(cfg.SOLVER.SEED);np.random.seed(cfg.SOLVER.SEED)
        torch.manual_seed(cfg.SOLVER.SEED);torch.cuda.manual_seed_all(cfg.SOLVER.SEED)
    seed();torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    output=Path(cfg.OUTPUT_DIR)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Choose an empty OUTPUT_DIR')
    loader,_,val,nq,nc,cameras,views=make_dataloader(cfg);loader,dataset=indexed_loader(loader)
    if (len(dataset),nq,len(val.dataset)-nq,nc)!=(15514,1394,26111,500):raise ValueError('Unexpected official NightReID split')
    model=make_model(cfg,nc,cameras,views,initialize=False)
    # load_baseline loads all backbone, MLDC, BN and classifier weights strictly.
    model.load_baseline(str(checkpoint))
    cp=torch.load(str(checkpoint),map_location='cpu',mmap=True,weights_only=False)
    if cp.get('training_dataset_fingerprint')!=dataset.fingerprint:
        raise ValueError('Baseline training image order/identity mapping differs from current data')
    initial_epoch=int(cp.get('epoch',0));del cp
    if cfg.NFC.FREEZE_ID_CLASSIFIER:model.classifier.requires_grad_(False)
    model.to('cuda:0');output.mkdir(parents=True,exist_ok=True)
    logger=setup_logger('transreid',str(output),if_train=True);logger.info('Config:\n%s',cfg)
    kd=build_fixed_targets(cfg,model,dataset,val.dataset.transform,torch.device('cuda:0'),logger)
    seed()  # Teacher extraction must not consume the student's sampler/augmentation seed.
    metadata=dict(variant=model.architecture()['variant'],architecture=model.architecture(),config=cfg.dump(),
        initialization=str(checkpoint),source_epoch=initial_epoch,epochs=cfg.SOLVER.MAX_EPOCHS,
        training_dataset_fingerprint=dataset.fingerprint,teacher_targets='nfc_teacher_targets.pth',
        loss='ID+Triplet+0.1 CNL+NFC.WEIGHT*T^2 KL(teacher class probability || student)',
        teacher_refresh='fixed offline before first step',resume_supported=False,
        trainable='original ViT+MLDC+BN; classifier frozen by default; no extra student head',
        inference='original single-image model; no NFC, prototypes, or teacher')
    (output/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    loss_fn,center=make_loss(cfg,nc);optimizer,_=make_optimizer(cfg,model,center)
    configure_head_learning_rates(cfg,model,optimizer)
    scheduler=wrap_weight_scheduler(cfg,model,optimizer,create_fixed_equal_scheduler(cfg,optimizer))
    do_train(cfg,model,loader,val,optimizer,scheduler,loss_fn,nq,dataset,nfc_kd=kd)


if __name__=='__main__':main()
