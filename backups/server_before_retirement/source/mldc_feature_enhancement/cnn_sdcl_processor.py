"""Global MLDC/CNL training; legacy CNN three-readout evaluation remains supported."""
import gc
import json
import logging
import time
from pathlib import Path
import torch
from utils.metrics import R1_mAP_eval
from .processor import append_csv
from .cppi_diagnostics import query_statistics,summarize_query_changes
from .cppi_optim import lr_summary
from .global_cnl import create_global_memory,global_training_loss


@torch.no_grad()
def evaluate(cfg,model,loader,nq,device,query_details=True):
    model.eval()
    names=['global','shallow','concat'] if model.enabled else ['global']
    evaluators={name:R1_mAP_eval(nq,max_rank=50,feat_norm='yes',reranking=False) for name in names}
    for evaluator in evaluators.values():evaluator.reset()
    count=0
    for images,pids,camids,camera,view,_ in loader:
        data=model(images.to(device),cam_label=camera.to(device),view_label=view.to(device),diagnostics=True)
        features={'global':data['global_unit'],'concat':data['retrieval']}
        if model.enabled:features['shallow']=data['shallow_unit']
        for name,evaluator in evaluators.items():
            evaluator.update((features[name].cpu(),pids,camids))
        count+=len(images)
    metrics={};queries={}
    for name,evaluator in evaluators.items():
        cmc,mean_ap,distance,*_=evaluator.compute()
        metrics[name]=dict(mAP=float(mean_ap),rank1=float(cmc[0]),
            rank5=float(cmc[min(4,len(cmc)-1)]),rank10=float(cmc[min(9,len(cmc)-1)]))
        if query_details:queries[name]=query_statistics(distance,evaluator.pids,evaluator.camids,nq)
        del distance
        evaluator.reset()
    diagnostics=dict(images=count,primary_readout='concat' if model.enabled else 'global',
        note='Same-checkpoint readout comparison; independent baseline training is a separate control')
    if query_details:
        diagnostics['query_statistics']=queries
        if model.enabled:
            changes={}
            for reference in ('global','shallow'):
                row=summarize_query_changes(queries['concat'],queries[reference])
                row['note']='Concatenation versus '+reference+' in the same checkpoint'
                changes['concat_vs_'+reference]=row
            valid=[i for i,row in enumerate(queries['global']) if row['valid']]
            union=[i for i in valid if queries['global'][i]['rank1'] or queries['shallow'][i]['rank1']]
            changes['branch_union_correct']=len(union)
            changes['concat_retained_union_correct']=sum(queries['concat'][i]['rank1'] for i in union)
            changes['concat_lost_union_query_indices']=[i for i in union if not queries['concat'][i]['rank1']]
            changes['concat_new_correct_query_indices']=[i for i in valid if
                queries['concat'][i]['rank1'] and not queries['global'][i]['rank1'] and not queries['shallow'][i]['rank1']]
            diagnostics['query_changes']=changes
    gc.collect()
    return metrics,diagnostics


def save_checkpoint(path,model,cfg,epoch,metrics,dataset_fingerprint=None,memory=None):
    pending=path.with_suffix('.pending')
    torch.save(dict(variant=model.architecture()['variant'],architecture=model.architecture(),epoch=epoch,
        state_dict=model.state_dict(),config=cfg.dump(),metrics=metrics,
        training_dataset_fingerprint=dataset_fingerprint,
        cnl_memory=memory.state_dict() if memory is not None else None),pending)
    pending.replace(path)


def do_train(cfg,model,train_loader,val_loader,optimizer,scheduler,nq,dataset):
    if model.mode!='mldc_cnl' or model.enabled:
        raise ValueError('The current training path uses one global descriptor without a shallow branch')
    device=torch.device('cuda:0');model.to(device)
    logger=logging.getLogger('transreid.train');output=Path(cfg.OUTPUT_DIR)
    memory=create_global_memory(cfg,model,dataset,device)
    scaler=torch.amp.GradScaler('cuda');best=-1.
    for epoch in range(1,cfg.SOLVER.MAX_EPOCHS+1):
        model.train();started=time.monotonic();count=batches=updates=0
        totals={};cnl_totals={}
        for iteration,(images,labels,cameras,views,indices) in enumerate(train_loader,1):
            images,labels,cameras,views,indices=[value.to(device) for value in (images,labels,cameras,views,indices)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                data=model(images,labels,cam_label=cameras,view_label=views)
                loss,detail,stats=global_training_loss(data,labels,cfg,memory,indices,cameras)
            if not torch.isfinite(loss).item():raise FloatingPointError('Non-finite global MLDC/CNL loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            if epoch==1 and iteration==1:
                prefixes=['base.patch_embed.','base.blocks.','bottleneck.','classifier.']
                gradients={prefix:sum(float(p.grad.float().square().sum()) for name,p in model.named_parameters()
                    if name.startswith(prefix) and p.grad is not None)**.5 for prefix in prefixes}
                logger.info('First-batch joint gradient norms: %s',gradients)
                (output/'startup_gradients.json').write_text(json.dumps(gradients,indent=2))
            scale_before=scaler.get_scale()
            scaler.step(optimizer);scaler.update()
            applied=scaler.get_scale()>=scale_before
            if applied and memory is not None:
                memory.update(data['global_unit'],indices,labels,cameras)
            batches+=1;updates+=int(applied);size=len(labels);count+=size
            detail.update(loss=float(loss.detach()),
                global_accuracy=float((data['scores'].argmax(1)==labels).float().mean()))
            for key,value in detail.items():totals[key]=totals.get(key,0.)+value*size
            for key in ('anchors','active_anchors','empty_positive_sets','selected_positives','available_positives'):
                cnl_totals[key]=cnl_totals.get(key,0)+stats.get(key,0)
            if iteration==1 or iteration%cfg.SOLVER.LOG_PERIOD==0:
                logger.info('Epoch[%d] Iter[%d/%d] loss %.5f CNL %.5f updates %d/%d LR %s neighbors %s',
                    epoch,iteration,len(train_loader),totals['loss']/count,totals['cnl']/count,updates,batches,
                    lr_summary(optimizer),stats)
        if count==0:raise RuntimeError('Empty training loader')
        scheduler.step()
        row=dict(epoch=epoch,**{key:value/count for key,value in totals.items()},batches=batches,
            optimizer_updates=updates,amp_skipped_steps=batches-updates,seconds=time.monotonic()-started)
        row.update(cnl_active_fraction=cnl_totals['active_anchors']/max(cnl_totals['anchors'],1),
            cnl_empty_fraction=cnl_totals['empty_positive_sets']/max(cnl_totals['anchors'],1),
            cnl_positive_coverage=cnl_totals['selected_positives']/max(cnl_totals['available_positives'],1),
            memory_images=int(memory.valid.sum()) if memory is not None else 0)
        append_csv(output/'training.csv',row)
        logger.info('Epoch %d complete: %s',epoch,row)
        metrics=None
        if epoch%cfg.SOLVER.EVAL_PERIOD==0 or epoch==cfg.SOLVER.MAX_EPOCHS:
            metrics,diagnostics=evaluate(cfg,model,val_loader,nq,device)
            for readout,values in metrics.items():
                append_csv(output/'evaluation.csv',dict(epoch=epoch,readout=readout,**values))
            with (output/'evaluation_diagnostics.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(epoch=epoch,metrics=metrics,**diagnostics))+'\n')
            logger.info('Global-readout evaluation epoch %d: %s',epoch,metrics)
            primary=diagnostics['primary_readout']
            if metrics[primary]['mAP']>best:
                best=metrics[primary]['mAP']
                save_checkpoint(output/'transformer_best.pth',model,cfg,epoch,metrics,dataset.fingerprint,memory=memory)
        if epoch%cfg.SOLVER.CHECKPOINT_PERIOD==0 or epoch==cfg.SOLVER.MAX_EPOCHS:
            save_checkpoint(output/f'transformer_{epoch}.pth',model,cfg,epoch,metrics,dataset.fingerprint,memory=memory)
