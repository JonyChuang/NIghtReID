"""Explicit training entry calls this module; importing it never starts training."""
import csv,json,logging,os,time
from pathlib import Path
import torch
from .model import VARIANT,BRANCHES,objectives,loss_weights,concat_loss_weight
from .evaluate import evaluate
from .optim import lr_summary


def append_csv(path,row):
    with Path(path).open('a',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if f.tell()==0:writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path,model,cfg,epoch,metrics):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    torch.save(dict(variant=VARIANT,epoch=epoch,config=cfg.dump(),architecture=model.architecture(),
        metrics=metrics,state_dict={name:p.detach().cpu() for name,p in model.state_dict().items()}),temp)
    os.replace(temp,path)


def best_link(source,target):
    """One epoch snapshot, three best pointers; avoid duplicate 350MB copies."""
    source=Path(source);target=Path(target);temporary=target.with_name(target.name+'.new')
    if temporary.exists():temporary.unlink()
    os.link(source,temporary);os.replace(temporary,target)


def gradient_summary(model):
    prefixes=['base.patch_embed.','base.blocks.0.','base.blocks.11.','base.norm.','bottleneck.','classifier.','last_bottleneck.','last_classifier.','concat_bottleneck.','concat_classifier.']
    return {prefix:sum(float(p.grad.float().square().sum()) for name,p in model.named_parameters()
        if name.startswith(prefix) and p.grad is not None)**.5 for prefix in prefixes}


def do_train(cfg,model,train_loader,val_loader,optimizer,scheduler,nq):
    device=torch.device('cuda:0');model.to(device);output=Path(cfg.OUTPUT_DIR)
    logger=logging.getLogger('transreid.train');scaler=torch.amp.GradScaler('cuda')
    best={name:-1. for name in BRANCHES}
    wg,wc=loss_weights(cfg);wz=concat_loss_weight(cfg)
    for epoch in range(1,cfg.SOLVER.MAX_EPOCHS+1):
        model.train();started=time.monotonic();count=updates=batches=0;total=0.
        sums={branch+'_'+term:0. for branch in BRANCHES for term in ('id','triplet')}
        correct={name:0 for name in BRANCHES}
        for iteration,(images,labels,camera,view) in enumerate(train_loader,1):
            images=images.to(device);labels=labels.to(device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                data=model(images,labels,cam_label=camera.to(device),view_label=view.to(device))
                loss,terms=objectives(data,labels,cfg)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            if iteration==1:
                detail=dict(epoch=epoch,lr=lr_summary(optimizer),gradient_norms=gradient_summary(model))
                with (output/'joint_gradients.jsonl').open('a') as f:f.write(json.dumps(detail)+'\n')
                logger.info('Joint branch gradients: %s',detail)
            before=model.classifier.weight._version
            scaler.step(optimizer);scaler.update();updates+=int(model.classifier.weight._version!=before);batches+=1
            size=len(labels);count+=size;total+=loss.item()*size
            for name,value in terms.items():sums[name]+=value.item()*size
            for name in correct:correct[name]+=(data['scores'][name].argmax(1)==labels).sum().item()
            if iteration==1 or iteration%cfg.SOLVER.LOG_PERIOD==0:
                logger.info('Epoch[%d] Iter[%d/%d] loss %.5f fusion(ID/Tri) %.5f/%.5f last(ID/Tri) %.5f/%.5f concat(ID/Tri) %.5f/%.5f acc %.4f/%.4f/%.4f updates %d LR %s',
                    epoch,iteration,len(train_loader),total/count,sums['fusion_id']/count,sums['fusion_triplet']/count,
                    sums['last_id']/count,sums['last_triplet']/count,sums['concat_id']/count,sums['concat_triplet']/count,
                    correct['fusion']/count,correct['last']/count,correct['concat']/count,updates,lr_summary(optimizer))
        if not count:raise RuntimeError('Empty training loader')
        scheduler.step(epoch)
        append_csv(output/'training.csv',dict(epoch=epoch,loss=total/count,**{k:v/count for k,v in sums.items()},
            fusion_loss_weight=wg,last_loss_weight=wc,concat_loss_weight=wz,
            weighted_fusion_loss=wg*(sums['fusion_id']+sums['fusion_triplet'])/count,
            weighted_last_loss=wc*(sums['last_id']+sums['last_triplet'])/count,
            weighted_concat_loss=wz*(sums['concat_id']+sums['concat_triplet'])/count,
            fusion_accuracy=correct['fusion']/count,last_accuracy=correct['last']/count,concat_accuracy=correct['concat']/count,batches=batches,
            optimizer_updates=updates,amp_skipped_steps=batches-updates,seconds=time.monotonic()-started))
        metrics=None
        if epoch%cfg.SOLVER.EVAL_PERIOD==0 or epoch==cfg.SOLVER.MAX_EPOCHS:
            metrics,comparisons=evaluate(cfg,model,val_loader,nq,device,output/'evaluation'/f'epoch_{epoch:03}')
            for name in BRANCHES:append_csv(output/'evaluation.csv',dict(epoch=epoch,branch=name,**metrics[name]))
            append_csv(output/'evaluation_concat.csv',dict(epoch=epoch,**metrics['concat']))
            with (output/'evaluation_diagnostics.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,metrics=metrics,comparisons=comparisons))+'\n')
            logger.info('Validation epoch %d: fusion=%s last=%s concat=%s; paired=%s',epoch,metrics['fusion'],metrics['last'],metrics['concat'],comparisons)
        save_due=metrics is not None or epoch%cfg.SOLVER.CHECKPOINT_PERIOD==0 or epoch==cfg.SOLVER.MAX_EPOCHS
        if save_due:
            snapshot=output/f'transformer_{epoch}.pth';save_checkpoint(snapshot,model,cfg,epoch,metrics)
            if metrics is not None:
                for name in BRANCHES:
                    if metrics[name]['mAP']>best[name]:
                        best[name]=metrics[name]['mAP'];best_link(snapshot,output/f'transformer_best_{name}.pth')
                        (output/f'best_{name}.json').write_text(json.dumps(dict(epoch=epoch,branch=name,metrics=metrics[name]),indent=2))
        logger.info('Epoch %d complete; optimizer updates %d/%d',epoch,updates,batches)
