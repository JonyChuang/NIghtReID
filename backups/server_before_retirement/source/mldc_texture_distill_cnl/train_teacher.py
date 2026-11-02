"""Stage A: train only the CNN teacher's identity-supervised encoder and heads."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config_file',default=str(Path(__file__).parent/'configs/texture_structure_nightreid.yml'))
    parser.add_argument('opts',nargs=argparse.REMAINDER); args=parser.parse_args()
    import numpy as np
    import torch
    from torch.nn import functional as F
    from config import cfg as defaults
    from datasets import make_dataloader
    from loss.triplet_loss import TripletLoss
    from utils.metrics import R1_mAP_eval
    from mldc_texture_distill_cnl.indexed_data import indexed_loader
    from mldc_texture_distill_cnl.model.make_model_mldc import add_mldc_config
    from mldc_texture_distill_cnl.model.texture_teacher import TextureTeacher
    cfg=add_mldc_config(defaults.clone()); cfg.merge_from_file(args.config_file); cfg.merge_from_list(args.opts); cfg.freeze()
    if cfg.SOLVER.SEED!=1234 or cfg.DATASETS.NAMES!='nightreid' or cfg.TEXTURE.TEACHER_EPOCHS!=40:
        raise ValueError('Use the matched seed1234 NightReID 40-epoch teacher protocol')
    if (list(cfg.INPUT.SIZE_TRAIN)!=[256,128] or list(cfg.INPUT.SIZE_TEST)!=[256,128]
            or list(cfg.MODEL.STRIDE_SIZE)!=[16,16] or not cfg.MODEL.STEM_CONV):
        raise ValueError('Use the aligned 256x128 CNN-stem, stride16 teacher protocol')
    os.environ['CUDA_VISIBLE_DEVICES']=cfg.MODEL.DEVICE_ID
    if not torch.cuda.is_available(): raise RuntimeError('Teacher training requires a visible CUDA GPU')
    random.seed(1234); np.random.seed(1234); torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=True
    train,_,val,nq,nc,_,_=make_dataloader(cfg); train,dataset=indexed_loader(train)
    assert (len(dataset),nq,len(val.dataset)-nq,nc)==(15514,1394,26111,500)
    output=Path(cfg.TEXTURE.TEACHER_PATH).parent
    if output.exists() and any(output.iterdir()): raise FileExistsError('Use an empty teacher output directory')
    teacher=TextureTeacher(nc,cfg.TEXTURE.BANK,(16,8),cfg.TEXTURE.DIM)
    teacher.initialize_stem(cfg.MODEL.PRETRAIN_PATH); teacher.cuda()
    frozen={name:t.detach().cpu().clone() for name,t in teacher.stem.state_dict().items()}
    optimizer=torch.optim.SGD([p for p in teacher.parameters() if p.requires_grad],
        lr=cfg.TEXTURE.TEACHER_LR,momentum=.9,weight_decay=1e-4)
    def factor(epoch):
        if epoch<5: return .01+.99*epoch/5
        return .002+.998*.5*(1+math.cos(math.pi*(epoch-5)/35))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    triplet=TripletLoss(margin=cfg.SOLVER.MARGIN)
    output.mkdir(parents=True,exist_ok=True); updates=0
    for epoch in range(1,41):
        teacher.train(); count=correct=0; total=0.
        for images,labels,_,_,_ in train:
            images=images.cuda(); labels=labels.cuda(); optimizer.zero_grad(set_to_none=True)
            result=teacher(images)
            loss=F.cross_entropy(result['logits'],labels)+triplet(result['raw'],labels)[0]
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite teacher objective')
            loss.backward(); optimizer.step(); updates+=1
            count+=len(labels); correct+=int((result['logits'].argmax(1)==labels).sum())
            total+=float(loss.detach())*len(labels)
        scheduler.step()
        with (output/'training.csv').open('a',newline='',encoding='utf-8') as handle:
            writer=csv.writer(handle)
            if handle.tell()==0: writer.writerow(['epoch','loss','accuracy','optimizer_updates'])
            writer.writerow([epoch,total/count,correct/count,updates])
        print(json.dumps(dict(epoch=epoch,loss=total/count,accuracy=correct/count)),flush=True)
    assert all(torch.equal(t.cpu(),frozen[n]) for n,t in teacher.stem.state_dict().items())
    # Fixed final-epoch evaluation; no teacher selection or tuning on official test IDs.
    teacher.eval(); evaluator=R1_mAP_eval(nq,max_rank=50,feat_norm='yes',reranking=False); evaluator.reset()
    with torch.inference_mode():
        for images,pids,camids,_,_,_ in val:
            evaluator.update((teacher(images.cuda())['unit'],pids,camids))
    cmc,mean_ap,*_=evaluator.compute()
    metadata=dict(stage='identity_texture_teacher_v1',trained=True,epoch=40,seed=1234,
        bank=cfg.TEXTURE.BANK,num_classes=nc,optimizer_updates=updates,
        training_dataset_fingerprint=dataset.fingerprint,training_images=len(dataset),
        stem_source=cfg.MODEL.PRETRAIN_PATH,stem_frozen=True,selected_by='fixed final epoch',
        metrics=dict(mAP=float(mean_ap),rank1=float(cmc[0])))
    torch.save(dict(state_dict=teacher.state_dict(),teacher_metadata=metadata,config=cfg.dump()),cfg.TEXTURE.TEACHER_PATH)
    (output/'teacher_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata),flush=True)


if __name__=='__main__': main()
