import csv
import gc
import json
import logging
import time
from pathlib import Path
import torch
from utils.metrics import R1_mAP_eval
from .model import objectives


def append_csv(path,row):
    with path.open('a',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if f.tell()==0:
            writer.writeheader()
        writer.writerow(row)


def save_patch_visualizations(cfg, samples, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    folder.mkdir(parents=True,exist_ok=True)
    for sample in samples:
        pixels=sample['image'].permute(1,2,0).numpy()
        pixels=(pixels*np.asarray(cfg.INPUT.PIXEL_STD)+np.asarray(cfg.INPUT.PIXEL_MEAN)).clip(0,1)
        layers=list(cfg.MODEL.PATCH_LAYERS)
        fig,axs=plt.subplots(len(layers),4,figsize=(7,3*len(layers)),squeeze=False,layout='constrained')
        for row,layer in enumerate(layers):
            axs[row,0].imshow(pixels);axs[row,0].set_title(f'Input / L{layer}');axs[row,0].axis('off')
            for col,(key,title) in enumerate([('patch_norm','Source patch norm'),('patch_change','Projected FEN correction'),('patch_attention','Compensation attention')],1):
                a=sample[key][row].numpy()
                axs[row,col].imshow(pixels)
                axs[row,col].imshow(a,cmap='turbo',alpha=.55,extent=(-.5,pixels.shape[1]-.5,pixels.shape[0]-.5,-.5),interpolation='bilinear')
                axs[row,col].set_title(f'{title}\n{a.min():.3g} - {a.max():.3g}');axs[row,col].axis('off')
        fig.suptitle(f'Query {sample["index"]}, PID {sample["pid"]}; per-panel scale; norms are not identity attribution')
        fig.savefig(folder/f'query_{sample["index"]:05d}.png',dpi=130);plt.close(fig)
    (folder/'definitions.json').write_text(json.dumps(dict(
        patch_layers=list(cfg.MODEL.PATCH_LAYERS),
        patch_norm='Raw post-block patch-token L2 over 768 channels',
        patch_change='Adapted FEN/SAFA output minus projected input, L2 over patch embedding channels',
        patch_attention='Cosine-softmax CLS-conditioned spatial reading weights; equal averaging over selected depths',
        color_scale='Each panel independently min-max normalized',samples=[dict(index=s['index'],pid=s['pid']) for s in samples]),indent=2))


@torch.no_grad()
def evaluate(cfg,model,loader,nq,device,visualization_dir=None):
    model.eval()
    evaluator=R1_mAP_eval(nq,max_rank=50,feat_norm=cfg.TEST.FEAT_NORM,reranking=False)
    evaluator.reset()
    delta_sum=input_sum=0.
    count=0
    patch_delta=patch_source=compensation_ratio=attention_entropy=0.
    samples=[]
    for images,pids,camids,camera,view,_ in loader:
        images=images.to(device)
        data=model(images,cam_label=camera.to(device),view_label=view.to(device),diagnostics=True)
        evaluator.update((data['retrieval'],pids,camids))
        delta_sum+=(data['enhanced'].float()-images.float()).flatten(1).norm(dim=-1).sum().item()
        input_sum+=images.float().flatten(1).norm(dim=-1).sum().item()
        if 'patch_attention' in data:
            import math
            attention=data['patch_attention'].float().flatten(2)
            attention_entropy+=float((-(attention*attention.clamp_min(1e-30).log()).sum(-1)/math.log(attention.shape[-1])).mean(1).sum())
            patch_delta+=float(data['patch_delta_norm'].mean(1).sum())
            patch_source+=float(data['patch_source_norm'].mean(1).sum())
            compensation_ratio+=float(((data['raw_readout'].float()-data['global_feature'].float()).norm(dim=-1)/data['global_feature'].float().norm(dim=-1).clamp_min(1e-12)).sum())
            if visualization_dir is not None and cfg.TEST.PATCH_VISUALIZE:
                for j in range(len(images)):
                    index=count+j
                    if index<nq and index in cfg.TEST.PATCH_VIS_INDICES:
                        sample=dict(index=index,pid=int(pids[j]),image=images[j].detach().cpu())
                        sample.update({key:data[key][j].detach().cpu() for key in ['patch_norm','patch_change','patch_attention']})
                        samples.append(sample)
        count+=len(images)
    cmc,mean_ap,distance,*_=evaluator.compute()
    metrics=dict(mAP=float(mean_ap),rank1=float(cmc[0]),rank5=float(cmc[4]),rank10=float(cmc[9]))
    diagnostics=dict(images=count,mean_input_correction_norm=delta_sum/max(count,1),
                     correction_norm_over_input_norm=delta_sum/max(input_sum,1e-12),
                     note='Correction magnitude is not a quality or identity gain score')
    if model.mode=='patch_feature':
        diagnostics.update(patch_enhancer_enabled=model.enabled,
                           patch_correction_over_projected_input_norm=patch_delta/max(patch_source,1e-12),
                           mean_compensation_over_global_norm=compensation_ratio/max(count,1),
                           mean_patch_attention_entropy=attention_entropy/max(count,1))
        if samples:
            save_patch_visualizations(cfg,samples,Path(visualization_dir))
    del distance,evaluator
    gc.collect()
    return metrics,diagnostics


def save_checkpoint(path,model,cfg,epoch,metrics):
    pending=path.with_suffix('.pending')
    torch.save(dict(variant=model.architecture()['variant'],architecture=model.architecture(),epoch=epoch,
                    state_dict=model.state_dict(),config=cfg.dump(),metrics=metrics),pending)
    pending.replace(path)


def do_train(cfg,model,train_loader,val_loader,optimizer,scheduler,nq):
    device=torch.device('cuda:0')
    model.to(device)
    logger=logging.getLogger('transreid.train')
    output=Path(cfg.OUTPUT_DIR)
    scaler=torch.amp.GradScaler('cuda')
    best=-1.
    for epoch in range(1,cfg.SOLVER.MAX_EPOCHS+1):
        model.train()
        assert all(b.training for b in model.base.blocks)
        start=time.monotonic()
        loss_sum=id_sum=tri_sum=correct=count=updates=batches=0
        for iteration,(images,labels,camera,view) in enumerate(train_loader,1):
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                data=model(images,labels,cam_label=camera.to(device),view_label=view.to(device))
                loss,identity,triplet=objectives(data,labels,cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if iteration==1 and epoch==1:
                norms={prefix:sum(float(p.grad.float().square().sum()) for n,p in model.named_parameters()
                                 if n.startswith(prefix) and p.grad is not None)**.5
                       for prefix in ['feat_enhancer.','patch_compensation.','base.patch_embed.','base.blocks.','classifier.']}
                logger.info('Joint objective gradient norms: %s',norms)
            before=model.classifier.weight._version
            scaler.step(optimizer);scaler.update()
            updates+=int(model.classifier.weight._version!=before)
            batches+=1
            size=len(labels);count+=size
            loss_sum+=loss.item()*size;id_sum+=identity.item()*size;tri_sum+=triplet.item()*size
            correct+=(data['scores'].argmax(1)==labels).sum().item()
            if iteration==1 or iteration%cfg.SOLVER.LOG_PERIOD==0:
                logger.info('Epoch[%d] Iter[%d/%d] Loss %.5f ID %.5f Tri %.5f Acc %.4f updates %d LR %.3g',epoch,iteration,len(train_loader),loss_sum/count,id_sum/count,tri_sum/count,correct/count,updates,optimizer.param_groups[0]['lr'])
        if count==0:
            raise RuntimeError('Empty training loader')
        scheduler.step(epoch)
        row=dict(epoch=epoch,loss=loss_sum/count,identity=id_sum/count,triplet=tri_sum/count,
                 accuracy=correct/count,batches=batches,optimizer_updates=updates,
                 amp_skipped_steps=batches-updates,seconds=time.monotonic()-start)
        append_csv(output/'training.csv',row)
        append_csv(output/'layer_weights.csv',dict(epoch=epoch,**{f'layer_{i+1}':float(w) for i,w in enumerate(model.readout_weights())}))
        logger.info('Epoch %d complete: %s',epoch,row)
        metrics=None
        if epoch%cfg.SOLVER.EVAL_PERIOD==0 or epoch==cfg.SOLVER.MAX_EPOCHS:
            metrics,diagnostics=evaluate(cfg,model,val_loader,nq,device,output/'visualizations'/f'epoch_{epoch:03d}')
            append_csv(output/'evaluation.csv',dict(epoch=epoch,**metrics))
            with (output/'evaluation_diagnostics.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(epoch=epoch,metrics=metrics,**diagnostics))+'\n')
            logger.info('Validation epoch %d: %s; %s',epoch,metrics,diagnostics)
            if metrics['mAP']>best:
                best=metrics['mAP']
                save_checkpoint(output/'transformer_best.pth',model,cfg,epoch,metrics)
        if epoch%cfg.SOLVER.CHECKPOINT_PERIOD==0 or epoch==cfg.SOLVER.MAX_EPOCHS:
            save_checkpoint(output/f'transformer_{epoch}.pth',model,cfg,epoch,metrics)
