"""Joint training and paired branch diagnostics for the ViTMatte transfer."""
import csv
import json
import logging
import time
from pathlib import Path
import torch
from torch.cuda import amp
from utils.metrics import R1_mAP_eval
from ..global_cnl import GlobalCNL
from ..objectives import supervised_losses
from ..query_diagnostics import query_statistics


def append_row(path,row):
    with Path(path).open('a',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row))
        if stream.tell()==0:writer.writeheader()
        writer.writerow(row)


def save_checkpoint(model,cfg,epoch,path,memory=None,metrics=None,dataset_fingerprint=None):
    torch.save(dict(state_dict=model.state_dict(),epoch=epoch,architecture=model.architecture(),
        config=cfg.dump(),metrics=metrics,cnl_memory=memory.state_dict() if memory is not None else None,
        training_dataset_fingerprint=dataset_fingerprint),path)


@torch.no_grad()
def evaluate(cfg,model,loader,num_query,device):
    model.eval()
    names=['fused','global']+(['detail'] if model.detail_enabled else [])
    evaluators={n:R1_mAP_eval(num_query,max_rank=50,feat_norm='yes',reranking=False) for n in names}
    for evaluator in evaluators.values():evaluator.reset()
    for images,pids,camids,camera,view,_ in loader:
        data=model(images.to(device),cam_label=camera.to(device),view_label=view.to(device),return_details=True)
        for name,evaluator in evaluators.items():
            evaluator.update((data[name+'_unit'],pids,camids))
    results,queries={},{}
    for name,evaluator in evaluators.items():
        cmc,mean_ap,distance,*_=evaluator.compute()
        results[name]=dict(mAP=float(mean_ap),rank1=float(cmc[0]),rank5=float(cmc[4]),rank10=float(cmc[9]))
        queries[name]=query_statistics(distance,evaluator.pids,evaluator.camids,num_query)
        del distance
    valid=[(g,f) for g,f in zip(queries['global'],queries['fused']) if g['valid'] and f['valid']]
    corrected=sum(not g['rank1'] and f['rank1'] for g,f in valid)
    broken=sum(g['rank1'] and not f['rank1'] for g,f in valid)
    diagnostics=dict(readouts=results,corrected=corrected,broken=broken,net_correct=corrected-broken,
        queries=queries,note='Same-checkpoint branch diagnostics; independent baseline training remains a separate control.')
    return results['fused'],diagnostics


def do_train(cfg,model,train_loader,val_loader,optimizer,scheduler,loss_fn,num_query,dataset):
    device=torch.device('cuda:0');model.to(device)
    output=Path(cfg.OUTPUT_DIR);output.mkdir(parents=True,exist_ok=True)
    logger=logging.getLogger('transreid.train')
    memory=(GlobalCNL([int(r[1]) for r in dataset.entries],[int(r[2]) for r in dataset.entries],dim=768,
        gamma=cfg.SOLVER.CNL_GAMMA,temperature=cfg.SOLVER.CNL_TEMPERATURE,
        momentum=cfg.SOLVER.CNL_MOMENTUM).to(device) if cfg.SOLVER.CNL_WEIGHT>0 else None)
    scaler=amp.GradScaler();best_map=-1.
    append_row(output/'layer_weights.csv',dict(epoch=0,**{
        'layer_'+str(i+1):float(v) for i,v in enumerate(model.readout_weights().detach().cpu())}))
    for epoch in range(1,cfg.SOLVER.MAX_EPOCHS+1):
        start=time.monotonic();model.train()
        sums={};samples=batches=updates=0;cnl_stats={}
        for iteration,(images,targets,camera,view,indices) in enumerate(train_loader,1):
            images,targets=images.to(device),targets.to(device)
            camera,view,indices=camera.to(device),view.to(device),indices.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():
                data=model(images,targets,cam_label=camera,view_label=view,return_training=True)
                supervised,terms=supervised_losses(data,targets,camera,loss_fn,cfg)
                if memory is not None:cnl,stats=memory(data['global_unit'],indices,targets,camera)
                else:cnl,stats=supervised*0.,{}
                loss=supervised+cfg.SOLVER.CNL_WEIGHT*cnl
            if not torch.isfinite(loss).item():raise FloatingPointError('Non-finite ViTMatte transfer loss')
            scaler.scale(loss).backward();previous_scale=scaler.get_scale()
            scaler.step(optimizer);scaler.update()
            applied=scaler.get_scale()>=previous_scale
            if applied and memory is not None:memory.update(data['global_unit'],indices,targets,camera)
            batches+=1;updates+=int(applied);size=len(targets);samples+=size
            for key,value in dict(terms,loss=float(loss.detach()),cnl=float(cnl.detach()),
                global_accuracy=float((data['global_scores'].argmax(1)==targets).float().mean()),
                fused_accuracy=float((data.get('fused_scores',data['global_scores']).argmax(1)==targets).float().mean())).items():
                sums[key]=sums.get(key,0.)+value*size
            for key in ['anchors','active_anchors','selected_positives','available_positives']:
                cnl_stats[key]=cnl_stats.get(key,0)+stats.get(key,0)
            if iteration==1 or iteration%cfg.SOLVER.LOG_PERIOD==0:
                logger.info('Epoch[%d] Iter[%d] loss %.4f, global ID/Tri %.4f/%.4f, fused ID/Tri %.4f/%.4f, CNL %.5f, updates %d/%d',
                    epoch,iteration,float(loss.detach()),terms['global_id'],terms['global_triplet'],
                    terms['fused_id'],terms['fused_triplet'],float(cnl.detach()),updates,batches)
        row=dict(epoch=epoch,**{k:v/samples for k,v in sums.items()},batches=batches,optimizer_updates=updates,
            amp_skipped_steps=batches-updates,seconds=time.monotonic()-start,
            cnl_active_fraction=cnl_stats.get('active_anchors',0)/max(cnl_stats.get('anchors',0),1),
            cnl_positive_coverage=cnl_stats.get('selected_positives',0)/max(cnl_stats.get('available_positives',0),1),
            memory_images=int(memory.valid.sum()) if memory is not None else 0)
        append_row(output/'training.csv',row)
        scheduler.step()
        append_row(output/'layer_weights.csv',dict(epoch=epoch,**{
            'layer_'+str(i+1):float(v) for i,v in enumerate(model.readout_weights().detach().cpu())}))
        metrics=None
        if epoch%cfg.SOLVER.EVAL_PERIOD==0:
            metrics,diagnostics=evaluate(cfg,model,val_loader,num_query,device)
            append_row(output/'evaluation.csv',dict(epoch=epoch,**metrics,
                corrected=diagnostics['corrected'],broken=diagnostics['broken'],net_correct=diagnostics['net_correct']))
            for name,values in diagnostics['readouts'].items():
                append_row(output/'branch_metrics.csv',dict(epoch=epoch,readout=name,**values))
            with (output/'evaluation_diagnostics.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(epoch=epoch,**diagnostics))+'\n')
            logger.info('Validation epoch %d: %s; corrected/broken/net %d/%d/%d',epoch,diagnostics['readouts'],
                diagnostics['corrected'],diagnostics['broken'],diagnostics['net_correct'])
            if metrics['mAP']>best_map:
                best_map=metrics['mAP']
                save_checkpoint(model,cfg,epoch,output/'transformer_best.pth',memory,metrics,dataset.fingerprint)
                (output/'best_metrics.json').write_text(json.dumps(dict(epoch=epoch,**metrics),indent=2))
        if epoch%cfg.SOLVER.CHECKPOINT_PERIOD==0 or epoch==cfg.SOLVER.MAX_EPOCHS:
            save_checkpoint(model,cfg,epoch,output/('transformer_'+str(epoch)+'.pth'),memory,metrics,dataset.fingerprint)


def do_inference(cfg,model,loader,num_query):
    device=torch.device(cfg.MODEL.DEVICE);model.to(device)
    return evaluate(cfg,model,loader,num_query,device)
