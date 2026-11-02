"""Joint supervised CNN/global training without constructing a CNL memory."""
import json
import logging
import time
from pathlib import Path
import torch
from .cnn_sdcl import supervised_objectives
from .cnn_sdcl_processor import evaluate,save_checkpoint
from .cppi_optim import lr_summary
from .processor import append_csv


def do_train(cfg,model,train_loader,val_loader,optimizer,scheduler,nq,dataset):
    if model.mode!='cnn_shallow_deep' or not model.enabled or cfg.SOLVER.CNL_WEIGHT!=0:
        raise ValueError('This training path requires the CNN branch and zero CNL')
    device=torch.device('cuda:0');model.to(device)
    logger=logging.getLogger('transreid.train');output=Path(cfg.OUTPUT_DIR)
    scaler=torch.amp.GradScaler('cuda');best=-1.
    for epoch in range(1,cfg.SOLVER.MAX_EPOCHS+1):
        model.train();started=time.monotonic();count=batches=updates=0;totals={}
        for iteration,(images,labels,cameras,views,_) in enumerate(train_loader,1):
            images,labels,cameras,views=[value.to(device) for value in (images,labels,cameras,views)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                data=model(images,labels,cam_label=cameras,view_label=views)
                loss,detail=supervised_objectives(data,labels,cfg)
            if not torch.isfinite(loss).item():raise FloatingPointError('Non-finite CNN/global loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            if epoch==1 and iteration==1:
                prefixes=['base.patch_embed.','base.blocks.','mldc.','cppi_pyramid.',
                    'cnn_reader.','bottleneck.','classifier.','shallow_bottleneck.','shallow_classifier.']
                gradients={prefix:sum(float(p.grad.float().square().sum()) for name,p in model.named_parameters()
                    if name.startswith(prefix) and p.grad is not None)**.5 for prefix in prefixes}
                logger.info('First-batch joint gradient norms: %s',gradients)
                (output/'startup_gradients.json').write_text(json.dumps(gradients,indent=2))
            scale_before=scaler.get_scale();scaler.step(optimizer);scaler.update()
            batches+=1;updates+=int(scaler.get_scale()>=scale_before);size=len(labels);count+=size
            detail.update(loss=float(loss.detach()),
                global_accuracy=float((data['scores'].argmax(1)==labels).float().mean()),
                shallow_accuracy=float((data['shallow_scores'].argmax(1)==labels).float().mean()))
            for key,value in detail.items():totals[key]=totals.get(key,0.)+value*size
            if iteration==1 or iteration%cfg.SOLVER.LOG_PERIOD==0:
                logger.info('Epoch[%d] Iter[%d/%d] loss %.5f weights %s LR %s',
                    epoch,iteration,len(train_loader),totals['loss']/count,
                    model.readout_weights()[8:].detach().cpu().tolist(),lr_summary(optimizer))
        if count==0:raise RuntimeError('Empty training loader')
        scheduler.step()
        row=dict(epoch=epoch,**{key:value/count for key,value in totals.items()},batches=batches,
            optimizer_updates=updates,amp_skipped_steps=batches-updates,seconds=time.monotonic()-started)
        row.update({f'weight_l{depth}':float(weight) for depth,weight in
            zip((9,10,11,12),model.readout_weights()[8:].detach().cpu())})
        append_csv(output/'training.csv',row)
        logger.info('Epoch %d complete: %s',epoch,row)
        metrics=None
        if epoch%cfg.SOLVER.EVAL_PERIOD==0 or epoch==cfg.SOLVER.MAX_EPOCHS:
            metrics,diagnostics=evaluate(cfg,model,val_loader,nq,device)
            for readout,values in metrics.items():
                append_csv(output/'evaluation.csv',dict(epoch=epoch,readout=readout,**values))
            with (output/'evaluation_diagnostics.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(epoch=epoch,metrics=metrics,**diagnostics))+'\n')
            logger.info('Global/shallow/concat evaluation epoch %d: %s',epoch,metrics)
            if metrics['concat']['mAP']>best:
                best=metrics['concat']['mAP']
                save_checkpoint(output/'transformer_best.pth',model,cfg,epoch,metrics,dataset.fingerprint)
        if epoch%cfg.SOLVER.CHECKPOINT_PERIOD==0 or epoch==cfg.SOLVER.MAX_EPOCHS:
            save_checkpoint(output/f'transformer_{epoch}.pth',model,cfg,epoch,metrics,dataset.fingerprint)
