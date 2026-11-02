"""Specialize the old small DCM ConvStream; this module never trains on import."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config_file',default=str(Path(__file__).parent/'configs/texture_relation_nightreid.yml'))
    parser.add_argument('opts',nargs=argparse.REMAINDER); args=parser.parse_args()
    import numpy as np
    import torch
    from torch.nn import functional as F
    from config import cfg as defaults
    from datasets import make_dataloader
    from loss.triplet_loss import TripletLoss
    from utils.metrics import R1_mAP_eval
    from mldc_convstream_texture_cnl.indexed_data import indexed_loader
    from mldc_convstream_texture_cnl.model.make_model_mldc import add_mldc_config
    from mldc_convstream_texture_cnl.model.texture_teacher import TextureTeacher,exposure_view,specialization_consistency
    cfg=add_mldc_config(defaults.clone()); cfg.merge_from_file(args.config_file); cfg.merge_from_list(args.opts); cfg.freeze()
    epochs=int(cfg.TEXTURE.TEACHER_EPOCHS); warmup=int(cfg.TEXTURE.TEACHER_WARMUP_EPOCHS)
    if (cfg.SOLVER.SEED!=1234 or cfg.DATASETS.NAMES!='nightreid' or epochs<1 or not 0<=warmup<epochs
            or not cfg.TEXTURE.ENABLED or list(cfg.INPUT.SIZE_TRAIN)!=[256,128]
            or list(cfg.INPUT.PIXEL_MEAN)!=[.5,.5,.5] or list(cfg.INPUT.PIXEL_STD)!=[.5,.5,.5]):
        raise ValueError('Use seed1234 NightReID and a fixed ConvStream specialization schedule')
    os.environ['CUDA_VISIBLE_DEVICES']=cfg.MODEL.DEVICE_ID
    if not torch.cuda.is_available(): raise RuntimeError('Teacher specialization requires a visible CUDA GPU')
    random.seed(1234); np.random.seed(1234); torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=True
    train,_,val,nq,nc,_,_=make_dataloader(cfg); train,dataset=indexed_loader(train)
    assert (len(dataset),nq,len(val.dataset)-nq,nc)==(15514,1394,26111,500)
    output=Path(cfg.TEXTURE.TEACHER_PATH).parent
    if output.exists() and any(output.iterdir()): raise FileExistsError('Use an empty teacher output directory')
    teacher=TextureTeacher(nc,cfg.TEXTURE.BANK,cfg.TEXTURE.GRID,cfg.TEXTURE.DIM)
    teacher.initialize_convstream(cfg.TEXTURE.TEACHER_INIT_PATH); teacher.cuda()
    bn_before={n:b.detach().cpu().clone() for n,b in teacher.convstream.named_buffers()}
    groups=[dict(params=list(teacher.convstream.parameters()),lr=cfg.TEXTURE.TEACHER_LR),
        dict(params=[p for name,p in teacher.named_parameters() if p.requires_grad and not name.startswith('convstream.')],
            lr=cfg.TEXTURE.TEACHER_LR*cfg.TEXTURE.TEACHER_HEAD_LR_FACTOR)]
    optimizer=torch.optim.SGD(groups,momentum=.9,weight_decay=1e-4)
    def factor(epoch):
        if epoch<warmup: return .01+.99*epoch/max(warmup,1)
        return .002+.998*.5*(1+math.cos(math.pi*min(epoch-warmup,epochs-warmup)/(epochs-warmup)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    triplet=TripletLoss(margin=cfg.SOLVER.MARGIN)
    output.mkdir(parents=True,exist_ok=True); updates=0; started=time.time()
    (output/'run_metadata.json').write_text(json.dumps(dict(stage='teacher_specialization',seed=1234,
        bank=cfg.TEXTURE.BANK,epochs=epochs,warmup_epochs=warmup,config=cfg.dump(),warmstart=teacher.provenance,
        training_dataset_fingerprint=dataset.fingerprint,training_started=True),indent=2),encoding='utf-8')
    for epoch in range(1,epochs+1):
        teacher.train(); count=correct=0; totals=dict(loss=0.,identity=0.,triplet=0.,illumination=0.,structure=0.)
        for iteration,(images,labels,_,_,_) in enumerate(train,1):
            images=images.cuda(); labels=labels.cuda(); optimizer.zero_grad(set_to_none=True)
            gain=images.new_empty((len(images),1,1,1)).uniform_(.7,1.3)
            gamma=images.new_empty((len(images),1,1,1)).uniform_(.8,1.2)
            first=teacher(images); second=teacher(exposure_view(images,gain,gamma))
            identity=.5*(F.cross_entropy(first['logits'],labels)+F.cross_entropy(second['logits'],labels))
            tri=.5*(triplet(first['raw'],labels)[0]+triplet(second['raw'],labels)[0])
            illumination,structure=specialization_consistency(first,second)
            loss=identity+tri+cfg.TEXTURE.TEACHER_ILLUM_WEIGHT*illumination+cfg.TEXTURE.TEACHER_STRUCTURE_WEIGHT*structure
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite teacher objective')
            loss.backward(); optimizer.step(); updates+=1
            size=len(labels); count+=size; correct+=int((first['logits'].argmax(1)==labels).sum())
            for key,value in dict(loss=loss,identity=identity,triplet=tri,illumination=illumination,structure=structure).items():
                totals[key]+=float(value.detach())*size
            if iteration==1 or iteration%50==0:
                progress=dict(stage='teacher_training',epoch=epoch,iteration=iteration,iterations=len(train),
                    loss=totals['loss']/count,accuracy=correct/count,optimizer_updates=updates,seed=1234,
                    elapsed_seconds=time.time()-started,gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                (output/'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
                print('TEACHER_PROGRESS '+json.dumps(progress),flush=True)
        scheduler.step(); row=dict(epoch=epoch,**{k:v/count for k,v in totals.items()},accuracy=correct/count,optimizer_updates=updates)
        with (output/'training.csv').open('a',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(row))
            if handle.tell()==0: writer.writeheader()
            writer.writerow(row)
        print(json.dumps(row),flush=True)
        # Small epoch snapshot retains completed work if the GPU instance stops.
        torch.save(dict(state_dict=teacher.state_dict(),epoch=epoch,optimizer_updates=updates,
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            random_state=random.getstate(),numpy_random_state=np.random.get_state(),
            torch_random_state=torch.get_rng_state(),cuda_random_state=torch.cuda.get_rng_state_all(),
            config=cfg.dump(),teacher_metadata=dict(teacher.provenance,trained=False,epoch=epoch,
                seed=1234,bank=cfg.TEXTURE.BANK,training_dataset_fingerprint=dataset.fingerprint)),
            output/'teacher_latest.pth')
    assert all(torch.equal(b.cpu(),bn_before[n]) for n,b in teacher.convstream.named_buffers())
    # One fixed-final report, without selecting a teacher epoch on the test split.
    teacher.eval(); evaluator=R1_mAP_eval(nq,max_rank=50,feat_norm='yes',reranking=False); evaluator.reset()
    with torch.inference_mode():
        for images,pids,camids,_,_,_ in val:
            evaluator.update((teacher(images.cuda())['unit'],pids,camids))
    cmc,mean_ap,*_=evaluator.compute()
    metadata=dict(teacher.provenance,stage='convstream_identity_texture_teacher_v2',trained=True,epoch=epochs,seed=1234,
        bank=cfg.TEXTURE.BANK,num_classes=nc,grid=list(cfg.TEXTURE.GRID),dim=cfg.TEXTURE.DIM,optimizer_updates=updates,
        training_dataset_fingerprint=dataset.fingerprint,training_images=len(dataset),selected_by='fixed final epoch',
        BN_statistics='retained from old DCM; convolution and small heads fine-tuned',
        metrics=dict(mAP=float(mean_ap),rank1=float(cmc[0])))
    torch.save(dict(state_dict=teacher.state_dict(),teacher_metadata=metadata,config=cfg.dump()),cfg.TEXTURE.TEACHER_PATH)
    (output/'teacher_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata),flush=True)


if __name__=='__main__': main()
