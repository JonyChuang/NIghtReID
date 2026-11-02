"""Explicit NightReID PAD training entry; importing never starts a run."""
import argparse,json,os,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    parser=argparse.ArgumentParser(description='MLDC+CNL with P12 prototype attention distillation')
    parser.add_argument('--config_file',default=str(Path(__file__).parent/'configs/pad_p12_nightreid.yml'))
    parser.add_argument('opts',nargs=argparse.REMAINDER)
    args=parser.parse_args()
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
    cfg=add_mldc_config(defaults.clone());cfg.merge_from_file(args.config_file);cfg.merge_from_list(args.opts)
    if (cfg.DATASETS.NAMES!='nightreid' or cfg.SOLVER.SEED!=1234 or cfg.SOLVER.MAX_EPOCHS!=120
            or cfg.SOLVER.WARMUP_EPOCHS!=20 or cfg.SOLVER.WARMUP_METHOD!='cosine'
            or cfg.MODEL.SIE_CAMERA or cfg.MODEL.SIE_VIEW or cfg.MODEL.PRETRAIN_CHOICE!='imagenet'
            or cfg.MODEL.MLDC_READOUT!='mldc' or list(cfg.MODEL.MLDC_LAYERS)!=[9,10,11,12]
            or cfg.MODEL.MLDC_INIT!='uniform' or cfg.TEST.NECK_FEAT!='after'
            or cfg.SOLVER.CNL_WEIGHT!=.1 or cfg.SOLVER.CNL_GAMMA!=.9):
        raise ValueError('Preserve the matched NightReID/SSL/seed1234/120-epoch MLDC+CNL protocol')
    cfg.freeze();os.environ['CUDA_VISIBLE_DEVICES']=cfg.MODEL.DEVICE_ID
    if cfg.MODEL.DEVICE!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('Training requires a visible CUDA GPU')
    random.seed(1234);np.random.seed(1234);torch.manual_seed(1234);torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=True
    output=Path(cfg.OUTPUT_DIR)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty output directory; old ConvStream results are not overwritten')
    loader,_,val,nq,nc,cameras,views=make_dataloader(cfg);loader,dataset=indexed_loader(loader)
    if nq!=1394 or len(val.dataset)-nq!=26111 or nc!=500:
        raise ValueError('Expected official NightReID split: 500 training IDs, 1394 queries, 26111 gallery')
    model=make_model(cfg,nc,cameras,views)
    output.mkdir(parents=True,exist_ok=True)
    logger=setup_logger('transreid',str(output),if_train=True);logger.info('Config:\n%s',cfg)
    metadata=dict(variant='mldc_pad_p12_20261029',architecture=model.architecture(),
        seed=1234,config=cfg.dump(),initialization=cfg.MODEL.PRETRAIN_PATH,
        training_dataset_fingerprint=dataset.fingerprint,training_images=len(dataset),
        primary_endpoint=120,descriptor_dim=768,resume_supported=False,
        loss='final ID+Triplet+0.1 CNL; auxiliary student/teacher prototype ID and per-layer per-head KL',
        prototype='single identity prototype; student bootstrap then teacher EMA after warmup',
        inference='student head retained, no labels or identity prototypes')
    (output/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    loss_fn,center=make_loss(cfg,nc)
    optimizer,_=make_optimizer(cfg,model,center);configure_head_learning_rates(cfg,model,optimizer)
    scheduler=wrap_weight_scheduler(cfg,model,optimizer,create_fixed_equal_scheduler(cfg,optimizer))
    do_train(cfg,model,loader,val,optimizer,scheduler,loss_fn,nq,dataset)

if __name__=='__main__':main()
