import csv
import gc
import json
import logging
import time
from pathlib import Path
import numpy as np
import torch
from utils.metrics import R1_mAP_eval
from .model import objectives, VARIANT


def append_csv(path,row):
    with path.open('a',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row))
        if f.tell()==0:
            w.writeheader()
        w.writerow(row)


@torch.no_grad()
def evaluate(cfg,model,loader,nq,device):
    model.eval()
    names = ('retrieval','global','local','f3','f5','f6') if model.enabled else ('retrieval','global')
    evaluators={n:R1_mAP_eval(nq,max_rank=50,feat_norm='yes',reranking=False) for n in names}
    for e in evaluators.values():
        e.reset()
    for images,pids,camids,camera,view,_ in loader:
        data=model(images.to(device),cam_label=camera.to(device),view_label=view.to(device),diagnostics=True)
        for name,e in evaluators.items():
            e.update((data[name],pids,camids))
    results, flags = {}, {}
    for name in names:
        e=evaluators.pop(name)
        cmc,mean_ap,distance,pids,cams,*_=e.compute()
        results[name]=dict(mAP=float(mean_ap),rank1=float(cmc[0]),rank5=float(cmc[4]),rank10=float(cmc[9]))
        if name in ('retrieval','global'):
            pids,cams=np.asarray(pids),np.asarray(cams)
            correct=np.full(nq,-1,dtype=np.int8)
            for i in range(nq):
                order=np.argsort(distance[i])
                keep=~((pids[nq:][order]==pids[i]) & (cams[nq:][order]==cams[i]))
                relevant=pids[nq:][order[keep]]==pids[i]
                if relevant.any():
                    correct[i]=int(relevant[0])
            flags[name]=correct
        del distance,e
        gc.collect()
    fixed=int(((flags['retrieval']==1)&(flags['global']==0)).sum())
    broken=int(((flags['retrieval']==0)&(flags['global']==1)).sum())
    return results,dict(rank1_fixed_by_local=fixed,rank1_broken_by_local=broken,
                        note='Global is the same joint checkpoint with shared BN; not an independent baseline')


def save_checkpoint(path,model,cfg,epoch,metrics):
    pending=path.with_suffix('.pending')
    torch.save(dict(variant=VARIANT,fft=True,architecture=model.architecture(),config=cfg.dump(),
                    epoch=epoch,state_dict=model.state_dict(),metrics=metrics),pending)
    pending.replace(path)


def do_train(cfg,model,train_loader,val_loader,optimizer,scheduler,nq):
    device=torch.device('cuda:0')
    model.to(device)
    log=logging.getLogger('transreid.train')
    output=Path(cfg.OUTPUT_DIR)
    scaler=torch.amp.GradScaler('cuda')
    best=-1.
    for epoch in range(1,cfg.SOLVER.MAX_EPOCHS+1):
        model.train()
        assert all(b.training for b in model.base.blocks)
        start=time.monotonic()
        total=total_id=total_tri=correct=local_correct=samples=0
        branch_sum=np.zeros(4 if model.enabled else 1)
        updates=0
        for iteration,(images,labels,camera,view) in enumerate(train_loader,1):
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                data=model(images,labels,cam_label=camera.to(device),view_label=view.to(device))
                loss,ids,tri=objectives(data,labels,cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if iteration==1 and epoch==1:
                def grad_norm(prefix):
                    return sum(float(p.grad.float().square().sum()) for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None)**.5
                log.info('First joint backward: backbone=%g LPAM=%g Q=%g K=%g V=%g (single-token Q/K expected zero)',grad_norm('base.blocks.'),grad_norm('lpam.'),grad_norm('lpam.conv_block.self_attn.query.'),grad_norm('lpam.conv_block.self_attn.key.'),grad_norm('lpam.conv_block.self_attn.value.'))
            before=model.classifier.weight._version
            scaler.step(optimizer);scaler.update()
            updates+=int(model.classifier.weight._version!=before)
            size=len(labels);samples+=size
            total+=loss.item()*size;total_id+=ids.mean().item()*size;total_tri+=tri.item()*size
            branch_sum+=ids.double().cpu().numpy()*size
            correct+=(data['scores'][0].argmax(1)==labels).sum().item()
            local_correct+=sum((s.argmax(1)==labels).sum().item() for s in data['scores'][1:])
            if iteration==1 or iteration%cfg.SOLVER.LOG_PERIOD==0:
                log.info('Epoch[%d] Iter[%d/%d] Loss %.5f ID %.5f Tri %.5f GlobalAcc %.4f updates %d LR %.3g',epoch,iteration,len(train_loader),total/samples,total_id/samples,total_tri/samples,correct/samples,updates,optimizer.param_groups[0]['lr'])
        if samples==0:
            raise RuntimeError('Empty training loader')
        scheduler.step(epoch)
        row=dict(epoch=epoch,loss=total/samples,identity=total_id/samples,triplet=total_tri/samples,
                 global_accuracy=correct/samples,local_accuracy=local_correct/(3*samples) if model.enabled else 0.,
                 optimizer_updates=updates,seconds=time.monotonic()-start)
        row.update({f'branch_{i}_id':float(v/samples) for i,v in enumerate(branch_sum)})
        append_csv(output/'training.csv',row)
        append_csv(output/'layer_weights.csv',dict(epoch=epoch,**{f'layer_{i+1}':float(v) for i,v in enumerate(model.readout_weights())}))
        log.info('Epoch %d complete: %s',epoch,row)
        metrics=None
        if epoch%cfg.SOLVER.EVAL_PERIOD==0 or epoch==cfg.SOLVER.MAX_EPOCHS:
            results,diagnostics=evaluate(cfg,model,val_loader,nq,device)
            metrics=results['retrieval']
            append_csv(output/'evaluation.csv',dict(epoch=epoch,**metrics))
            for name,values in results.items():
                append_csv(output/'branch_evaluation.csv',dict(epoch=epoch,readout=name,**values))
            with (output/'evaluation_diagnostics.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(epoch=epoch,readouts=results,**diagnostics))+'\n')
            log.info('Epoch %d validation: %s; %s',epoch,results,diagnostics)
            if metrics['mAP']>best:
                best=metrics['mAP']
                save_checkpoint(output/'transformer_best.pth',model,cfg,epoch,metrics)
        if epoch%cfg.SOLVER.CHECKPOINT_PERIOD==0 or epoch==cfg.SOLVER.MAX_EPOCHS:
            save_checkpoint(output/f'transformer_{epoch}.pth',model,cfg,epoch,metrics)
