"""Single-GPU training with epoch-wise MLDC weights and best-checkpoint saving."""
import csv
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda import amp
from utils.metrics import R1_mAP_eval
from .visualization import save_visualizations
from ..global_cnl import GlobalCNL
from ..query_diagnostics import query_statistics


def record_weights(model, epoch, output, logger):
    weights = model.readout_weights().detach().float().cpu().tolist()
    path = Path(output) / 'layer_weights.csv'
    write_header = epoch == 0 or not path.exists()
    with path.open('w' if epoch == 0 else 'a', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(['epoch'] + [f'layer_{i}' for i in range(1, 13)])
        writer.writerow([epoch] + weights)
    logger.info('Epoch[%s] %s layer weights: %s', epoch, model.readout,
                ', '.join(f'{i}:{weight:.5f}' for i, weight in enumerate(weights, 1)))


@torch.no_grad()
def evaluate(cfg, model, loader, num_query, device, include_diagnostics=False):
    model.eval()
    names = ['retrieval'] + (['raw_readout', 'raw_last_cls', 'raw_uniform_cls'] if include_diagnostics else [])
    evaluators = {name: R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
                                  reranking=cfg.TEST.RE_RANKING) for name in names}
    for evaluator in evaluators.values():
        evaluator.reset()
    norm_sum = torch.zeros(12, dtype=torch.float64)
    cosine_sum, count = 0., 0
    for images, pids, camids, camera, view, _ in loader:
        output, raw, layers = model(images.to(device), cam_label=camera.to(device),
                                   view_label=view.to(device), return_components=True)
        features = {'retrieval': output}
        if include_diagnostics:
            features.update(raw_readout=raw, raw_last_cls=layers[-1], raw_uniform_cls=layers.mean(0))
            norm_sum += layers.float().norm(dim=-1).sum(1).double().cpu()
            cosine_sum += F.cosine_similarity(raw.float(), layers[-1].float()).sum().item()
            count += len(images)
        for name, evaluator in evaluators.items():
            evaluator.update((features[name], pids, camids))
    results = {}
    queries = None
    for name, evaluator in evaluators.items():
        cmc, mean_ap, distance, *_ = evaluator.compute()
        if name == 'retrieval':
            queries = query_statistics(distance, evaluator.pids, evaluator.camids, num_query)
        del distance
        results[name] = {'mAP': float(mean_ap), 'rank1': float(cmc[0]),
                         'rank5': float(cmc[4]), 'rank10': float(cmc[9])}
    if not include_diagnostics:
        return results['retrieval']
    weights = model.readout_weights().detach().double().cpu()
    weighted_norms = weights * norm_sum / count
    diagnostics = {'query_statistics': {'global': queries}, 'retrieval_mode': getattr(model, 'retrieval', 'mldc'),
                   'training_feature': getattr(model, 'train_feature', 'mldc'),
                   'readouts': {name: results[name] for name in names[1:]},
                   'layer_weights': weights.tolist(), 'mean_raw_layer_norms': (norm_sum / count).tolist(),
                   'weighted_norm_share_proxy': (weighted_norms / weighted_norms.sum().clamp_min(1e-12)).tolist(),
                   'raw_readout_last_cls_cosine_mean': cosine_sum / count,
                   'note': 'Same checkpoint, before-BN diagnostics; selected CLS9-12 are refined when PAD is enabled. '
                           'Weighted norms are a scale proxy, not causal contributions.'}
    return results['retrieval'], diagnostics


def record_evaluation(output, epoch, metrics, diagnostics):
    with (Path(output) / 'evaluation.csv').open('a', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(['epoch', 'mAP', 'rank1', 'rank5', 'rank10'])
        writer.writerow([epoch] + [metrics[key] for key in ('mAP', 'rank1', 'rank5', 'rank10')])
    with (Path(output) / 'diagnostics.csv').open('a', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(['epoch', 'feature', 'mAP', 'rank1', 'rank5', 'rank10'])
        for name, row in diagnostics['readouts'].items():
            writer.writerow([epoch, name] + [row[key] for key in ('mAP', 'rank1', 'rank5', 'rank10')])
    with (Path(output) / 'evaluation_diagnostics.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({'epoch': epoch, 'retrieval': metrics, **diagnostics}) + '\n')


def save_checkpoint(model, cfg, epoch, path, metrics=None, memory=None, dataset_fingerprint=None,
                    prototype_memory=None, optimizer=None, scheduler=None, scaler=None):
    torch.save(dict(state_dict=model.state_dict(), epoch=epoch, architecture=model.architecture(),
        mldc_layers=list(model.mldc.selected_layers), train_feature=model.train_feature,
        readout=model.readout, temperature=model.mldc.temperature, config=cfg.dump(),
        metrics=metrics, variant=model.architecture()['variant'],
        training_dataset_fingerprint=dataset_fingerprint,
        cnl_memory=memory.state_dict() if memory is not None else None,
        cnl_threshold_reference=getattr(memory,'threshold_reference',None),
        prototype_memory=prototype_memory.state_dict() if prototype_memory is not None else None,
        optimizer=optimizer.state_dict() if optimizer is not None else None,
        scheduler=scheduler.state_dict() if scheduler is not None else None,
        scaler=scaler.state_dict() if scaler is not None else None),path)


def do_train(cfg,model,train_loader,val_loader,optimizer,scheduler,loss_fn,num_query,dataset,nfc_kd=None,nfc_center=None):
    from ..prototype_memory import PrototypeMemory
    from ..training_updates import commit_training_memories
    if nfc_kd is not None and nfc_center is not None:
        raise ValueError('Run class KL and batch center contrast as separate experiments')
    device=torch.device('cuda:0')
    model.to(device)
    logger=logging.getLogger('transreid.train')
    output=Path(cfg.OUTPUT_DIR);output.mkdir(parents=True,exist_ok=True)
    labels=[int(r[1]) for r in dataset.entries]
    cameras=[int(r[2]) for r in dataset.entries]
    memory=(GlobalCNL(labels,cameras,dim=768,gamma=cfg.SOLVER.CNL_GAMMA,
        temperature=cfg.SOLVER.CNL_TEMPERATURE,momentum=cfg.SOLVER.CNL_MOMENTUM,
        threshold_reference=cfg.SOLVER.get('CNL_THRESHOLD_REFERENCE','all_nonself')).to(device)
        if cfg.SOLVER.CNL_WEIGHT>0 else None)
    prototypes=(PrototypeMemory(labels,model.classifier.out_features,model.descriptor_dim,
        cfg.PAD.MOMENTUM,cfg.PAD.MIN_IMAGES,update_mode=cfg.PAD.UPDATE_MODE,
        amu_temperature=cfg.PAD.AMU_TEMPERATURE).to(device) if model.pad_enabled else None)
    scaler=amp.GradScaler();best_map=-1.
    if memory is not None:
        logger.info('CNL threshold reference: %s; gamma %.3f, weight %.3f',
                    memory.threshold_reference,memory.gamma,cfg.SOLVER.CNL_WEIGHT)
    record_weights(model,0,output,logger)
    if nfc_kd is not None and cfg.NFC.EVAL_INITIAL:
        initial=evaluate(cfg,model,val_loader,num_query,device)
        (output/'initial_metrics.json').write_text(json.dumps(initial,indent=2),encoding='utf-8')
        logger.info('Initial single-image baseline: %s',json.dumps(initial))
    if cfg.TEST.MLDC_VISUALIZE:
        save_visualizations(cfg,model,val_loader,num_query,0,logger)
    for epoch in range(1,cfg.SOLVER.MAX_EPOCHS+1):
        start=time.monotonic();model.train()
        sums={k:0. for k in ['loss','identity','triplet','cnl','pad','prototype','attention','student_id','teacher_id','residual_ratio','entropy']}
        samples=correct=batches=updates=teacher_anchors=anchors=0
        cnl_totals={}
        nfc_sum=nfc_entropy=nfc_unscaled=0.
        nc_sum=nc_weighted=0.;nc_totals={}
        level_sums={level:dict(kl=0.,entropy=0.) for level in model.feature_layers}
        amu_sum=0.;amu_samples=bootstrap_samples=0;amu_min=1.;amu_max=0.
        for iteration,(images,targets,camera,view,indices) in enumerate(train_loader,1):
            images,targets=images.to(device),targets.to(device)
            camera,view,indices=camera.to(device),view.to(device),indices.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():
                scores,features,unit,pad=model(images,targets,cam_label=camera,view_label=view,
                    return_cnl_features=True,return_distillation=True,sample_indices=indices,
                    prototype_memory=prototypes,epoch=epoch)
                original_loss=loss_fn(scores,features,targets,camera)
                cnl,stats=memory(unit,indices,targets,camera) if memory is not None else (original_loss*0.,{})
                pad_loss=model.distillation_loss(pad,epoch)
                nfc_loss,nfc_stats=(nfc_kd(unit,indices,
                    student_logits=scores if cfg.NFC.STUDENT_PROBABILITY=='student_classifier' else None)
                    if nfc_kd is not None else (unit.sum()*0.,{}))
                nc_loss,nc_stats=(nfc_center(unit,indices,targets,camera,epoch)
                    if nfc_center is not None else (unit.sum()*0.,{}))
                nc_weight=nfc_center.current_weight(epoch) if nfc_center is not None else 0.
                loss=(original_loss+cfg.SOLVER.CNL_WEIGHT*cnl+pad_loss
                    +(cfg.NFC.WEIGHT*nfc_loss if nfc_kd is not None else 0.)+nc_weight*nc_loss)
            if not torch.isfinite(loss).item():
                raise FloatingPointError('Non-finite MLDC+CNL+PAD objective')
            scaler.scale(loss).backward()
            scale_before=scaler.get_scale();scaler.step(optimizer);scaler.update()
            applied=scaler.get_scale()>=scale_before
            amu=commit_training_memories(applied,memory,prototypes,unit,pad,indices,targets,camera)
            if amu is not None:
                amu_sum+=amu['amu_mu_sum'];amu_samples+=amu['amu_samples']
                bootstrap_samples+=amu['bootstrap_samples']
                if amu['amu_samples']:
                    amu_min=min(amu_min,amu['amu_mu_min']);amu_max=max(amu_max,amu['amu_mu_max'])
            size=len(targets);samples+=size;batches+=1;updates+=int(applied)
            nfc_sum+=float(nfc_loss.detach())*size
            nfc_entropy+=nfc_stats.get('teacher_entropy',0.)*size
            nfc_unscaled+=nfc_stats.get('probability_kl',0.)*size
            nc_sum+=float(nc_loss.detach())*size;nc_weighted+=nc_weight*float(nc_loss.detach())*size
            for key,value in nc_stats.items():nc_totals[key]=nc_totals.get(key,0)+value
            correct+=int((scores.argmax(1)==targets).sum())
            values=dict(loss=float(loss.detach()),identity=float(loss_fn.last_terms['identity']),
                triplet=float(loss_fn.last_terms['triplet']),cnl=float(cnl.detach()),pad=float(pad_loss.detach()),
                prototype=float(pad['prototype'].detach()),attention=float(pad['attention'].detach()),
                student_id=float(pad['student_id'].detach()),teacher_id=float(pad['teacher_id'].detach()),
                residual_ratio=pad['stats'].get('residual_norm_ratio',0.),
                entropy=pad['stats'].get('student_attention_entropy',0.))
            for k,value in values.items():sums[k]+=value*size
            for level,totals in level_sums.items():
                totals['kl']+=pad['stats'].get('attention_kl_layer%d'%level,0.)*size
                totals['entropy']+=pad['stats'].get('attention_entropy_layer%d'%level,0.)*size
            teacher_anchors+=pad['stats']['teacher_anchors'];anchors+=pad['stats']['anchors']
            for k in ['anchors','active_anchors','empty_positive_sets','selected_positives','available_positives',
                      'legacy_empty_positive_sets','loss_candidate_empty_positive_sets',
                      'legacy_selected_positives','loss_candidate_selected_positives',
                      'reference_max_changed_anchors','reference_recovered_anchors']:
                cnl_totals[k]=cnl_totals.get(k,0)+stats.get(k,0)
            if iteration==1 or iteration%cfg.SOLVER.LOG_PERIOD==0:
                if nfc_center is not None:
                    if cfg.NFC_CENTER.LOSS=='mean_shift':
                        count=max(nc_totals.get('active_anchors',0),1)
                        logger.info('Epoch[%d] MeanShift raw %.8f weighted %.8f lambda %.4f active %.4f neighbors %.3f mean_norm %.5f drift %.5f',
                            epoch,nc_sum/samples,nc_weighted/samples,nc_weight,
                            nc_totals.get('active_anchors',0)/max(nc_totals.get('anchors',0),1),
                            nc_totals.get('support_images',0)/count,nc_totals.get('mean_norm_sum',0.)/count,
                            nc_totals.get('shift_norm_sum',0.)/count)
                    else:
                        logger.info('Epoch[%d] NFC center raw %.8f weighted %.8f lambda %.4f active %.4f pairs %d mean support %.3f',
                            epoch,nc_sum/samples,nc_weighted/samples,nc_weight,
                            nc_totals.get('active_anchors',0)/max(nc_totals.get('anchors',0),1),
                            nc_totals.get('positive_centers',0),
                            nc_totals.get('support_images',0)/max(nc_totals.get('positive_centers',0),1))
                if nfc_kd is not None:
                    logger.info('Epoch[%d] NFC identity KD T^2KL %.8f weighted %.8f raw KL %.8f teacher entropy %.5f',
                        epoch,nfc_sum/samples,cfg.NFC.WEIGHT*nfc_sum/samples,nfc_unscaled/samples,nfc_entropy/samples)
                if not model.pad_enabled:
                    logger.info('Epoch[%d] Iter[%d/%d] Loss %.5f ID %.5f Tri %.5f CNL %.5f updates %d/%d',
                        epoch,iteration,len(train_loader),sums['loss']/samples,sums['identity']/samples,
                        sums['triplet']/samples,sums['cnl']/samples,updates,batches)
                    if model.pem_enabled:
                        logger.info('Epoch[%d] PEM last-forward: %s',epoch,json.dumps(model.pem_diagnostics()))
                else:
                    logger.info('Epoch[%d] Iter[%d/%d] Loss %.5f ID %.5f Tri %.5f CNL %.5f PAD %.5f KL %.5f teacher %d/%d updates %d/%d',
                        epoch,iteration,len(train_loader),sums['loss']/samples,sums['identity']/samples,
                        sums['triplet']/samples,sums['cnl']/samples,sums['pad']/samples,sums['attention']/samples,
                        teacher_anchors,anchors,updates,batches)
                    logger.info('Epoch[%d] KL per level: L9 %.6f L10 %.6f L11 %.6f L12 %.6f',
                        epoch,*(level_sums[level]['kl']/samples for level in model.feature_layers))
                if memory is not None:
                    total=max(cnl_totals.get('anchors',0),1)
                    logger.info('Epoch[%d] CNL empty actual/legacy/candidate %.4f/%.4f/%.4f; recovered %.4f',
                        epoch,cnl_totals.get('empty_positive_sets',0)/total,
                        cnl_totals.get('legacy_empty_positive_sets',0)/total,
                        cnl_totals.get('loss_candidate_empty_positive_sets',0)/total,
                        cnl_totals.get('reference_recovered_anchors',0)/total)
        row=dict(epoch=epoch,global_id=sums['identity']/samples,global_triplet=sums['triplet']/samples,
            cnl=sums['cnl']/samples,loss=sums['loss']/samples,global_accuracy=correct/samples,
            batches=batches,optimizer_updates=updates,amp_skipped_steps=batches-updates,
            seconds=time.monotonic()-start,
            cnl_active_fraction=cnl_totals.get('active_anchors',0)/max(cnl_totals.get('anchors',0),1),
            cnl_empty_fraction=cnl_totals.get('empty_positive_sets',0)/max(cnl_totals.get('anchors',0),1),
            cnl_positive_coverage=cnl_totals.get('selected_positives',0)/max(cnl_totals.get('available_positives',0),1),
            memory_images=int(memory.valid.sum()) if memory is not None else 0,
            cnl_legacy_empty_fraction=cnl_totals.get('legacy_empty_positive_sets',0)/max(cnl_totals.get('anchors',0),1),
            cnl_loss_candidate_empty_fraction=cnl_totals.get('loss_candidate_empty_positive_sets',0)/max(cnl_totals.get('anchors',0),1),
            cnl_legacy_positive_coverage=cnl_totals.get('legacy_selected_positives',0)/max(cnl_totals.get('available_positives',0),1),
            cnl_loss_candidate_positive_coverage=cnl_totals.get('loss_candidate_selected_positives',0)/max(cnl_totals.get('available_positives',0),1),
            cnl_reference_max_changed_fraction=cnl_totals.get('reference_max_changed_anchors',0)/max(cnl_totals.get('anchors',0),1),
            cnl_reference_recovered_fraction=cnl_totals.get('reference_recovered_anchors',0)/max(cnl_totals.get('anchors',0),1),
            pad_weighted=sums['pad']/samples,pad_prototype=sums['prototype']/samples,
            pad_attention_kl=sums['attention']/samples,pad_student_id=sums['student_id']/samples,
            pad_teacher_id=sums['teacher_id']/samples,pad_teacher_coverage=teacher_anchors/max(anchors,1),
            pad_prototype_classes=int(prototypes.valid.sum()) if prototypes is not None else 0,
            pad_residual_norm_ratio=sums['residual_ratio']/samples,pad_attention_entropy=sums['entropy']/samples,
            pad_attention_weight=model.attention_weight(epoch),
            pad_amu_mu_mean=amu_sum/amu_samples if amu_samples else 0.,
            pad_amu_mu_min=amu_min if amu_samples else 0.,pad_amu_mu_max=amu_max if amu_samples else 0.,
            pad_amu_samples=amu_samples,pad_bootstrap_samples=bootstrap_samples)
        for level,totals in level_sums.items():
            row['pad_attention_kl_layer%d'%level]=totals['kl']/samples
            row['pad_attention_entropy_layer%d'%level]=totals['entropy']/samples
        if not model.pad_enabled:
            row={k:v for k,v in row.items() if not k.startswith('pad_')}
        if nfc_kd is not None:
            row.update(nfc_class_kl_T2=nfc_sum/samples,nfc_weighted=cfg.NFC.WEIGHT*nfc_sum/samples,
                       nfc_probability_kl=nfc_unscaled/samples,nfc_teacher_entropy=nfc_entropy/samples)
        if nfc_center is not None:
            if cfg.NFC_CENTER.LOSS=='mean_shift':
                count=max(nc_totals.get('active_anchors',0),1);support=max(nc_totals.get('support_images',0),1)
                row.update(nfc_ms_raw=nc_sum/samples,nfc_ms_weighted=nc_weighted/samples,nfc_ms_weight=nc_weight,
                    nfc_ms_active_fraction=nc_totals.get('active_anchors',0)/max(samples,1),
                    nfc_ms_mean_neighbors=nc_totals.get('support_images',0)/count,
                    nfc_ms_single_neighbor_fraction=nc_totals.get('single_neighbor_anchors',0)/count,
                    nfc_ms_multi_neighbor_fraction=nc_totals.get('multi_neighbor_anchors',0)/count,
                    nfc_ms_mean_target_norm=nc_totals.get('mean_norm_sum',0.)/count,
                    nfc_ms_mean_drift_norm=nc_totals.get('shift_norm_sum',0.)/count,
                    nfc_ms_mean_tangent_drift_norm=nc_totals.get('tangent_shift_norm_sum',0.)/count,
                    nfc_ms_mean_neighbor_cosine=nc_totals.get('neighbor_cosine_sum',0.)/count,
                    nfc_ms_mean_neighbor_variance=nc_totals.get('neighbor_variance_sum',0.)/count,
                    nfc_ms_graph_coverage=nc_totals.get('graph_active_images',0)/max(nc_totals.get('unique_images',0),1),
                    nfc_ms_graph_same_id_fraction=nc_totals.get('graph_same_id_edges',0)/max(nc_totals.get('graph_edges',0),1),
                    nfc_ms_graph_crosscam_fraction=nc_totals.get('graph_cross_camera_edges',0)/max(nc_totals.get('graph_edges',0),1),
                    nfc_ms_neighbor_same_id_fraction=nc_totals.get('neighbor_same_id_assignments',0)/support,
                    nfc_ms_neighbor_crosscam_fraction=nc_totals.get('neighbor_cross_camera_assignments',0)/support)
            else:
                row.update(nfc_center_raw=nc_sum/samples,nfc_center_weighted=nc_weighted/samples,
                    nfc_center_weight=nc_weight,nfc_center_active_fraction=nc_totals.get('active_anchors',0)/max(samples,1),
                    nfc_center_positive_centers=nc_totals.get('positive_centers',0),
                    nfc_center_mean_support=nc_totals.get('support_images',0)/max(nc_totals.get('positive_centers',0),1),
                    nfc_center_crosscam_positive_pairs=nc_totals.get('crosscam_positive_pairs',0),
                    nfc_center_graph_coverage=nc_totals.get('graph_active_images',0)/max(nc_totals.get('unique_images',0),1),
                    nfc_center_graph_same_id_fraction=nc_totals.get('graph_same_id_edges',0)/max(nc_totals.get('graph_edges',0),1),
                    nfc_center_graph_crosscam_fraction=nc_totals.get('graph_cross_camera_edges',0)/max(nc_totals.get('graph_edges',0),1),
                    nfc_center_mean_margin=nc_totals.get('margin_sum',0.)/max(nc_totals.get('positive_centers',0),1))
        if model.pem_enabled:
            for level,stats in model.pem_diagnostics().items():
                row['pem_gate_layer%d'%level]=stats['gate']
                for name in ['kept_fraction','cls_to_patch_kept_fraction','patch_to_cls_kept_fraction',
                             'cls_update_norm_ratio','patch_update_norm_ratio']:
                    row['pem_last_forward_%s_layer%d'%(name,level)]=stats.get(name)
        with (output/'training.csv').open('a',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(row))
            if f.tell()==0:writer.writeheader()
            writer.writerow(row)
        scheduler.step();record_weights(model,epoch,output,logger)
        metrics=None
        checkpoint_kwargs=dict(memory=memory,prototype_memory=prototypes,
            dataset_fingerprint=dataset.fingerprint,optimizer=optimizer,scheduler=scheduler,scaler=scaler)
        if nfc_kd is not None or nfc_center is not None:
            # This entry has no resume route; retain model/CNL without unused optimizer copies.
            checkpoint_kwargs.update(optimizer=None,scheduler=None,scaler=None)
        if epoch%cfg.SOLVER.EVAL_PERIOD==0:
            metrics,diagnostics=evaluate(cfg,model,val_loader,num_query,device,include_diagnostics=True)
            record_evaluation(output,epoch,metrics,diagnostics)
            logger.info('Validation epoch %d: mAP %.4f%% Rank-1 %.4f%%',epoch,100*metrics['mAP'],100*metrics['rank1'])
            if metrics['mAP']>best_map:
                best_map=metrics['mAP']
                save_checkpoint(model,cfg,epoch,output/'transformer_best.pth',metrics,**checkpoint_kwargs)
                (output/'best_metrics.json').write_text(json.dumps(dict(epoch=epoch,**metrics),indent=2),encoding='utf-8')
        if epoch%cfg.SOLVER.CHECKPOINT_PERIOD==0 or epoch==cfg.SOLVER.MAX_EPOCHS:
            save_checkpoint(model,cfg,epoch,output/('transformer_%d.pth'%epoch),metrics,**checkpoint_kwargs)
        if cfg.TEST.MLDC_VISUALIZE and (epoch%cfg.TEST.MLDC_VIS_PERIOD==0 or epoch==cfg.SOLVER.MAX_EPOCHS):
            save_visualizations(cfg,model,val_loader,num_query,epoch,logger)


def do_inference(cfg, model, val_loader, num_query):
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    metrics, diagnostics = evaluate(cfg, model, val_loader, num_query, device, include_diagnostics=True)
    logger = logging.getLogger('transreid.test')
    for name, value in metrics.items():
        logger.info('%s: %.4f%%', name, value * 100)
    for name, values in diagnostics['readouts'].items():
        logger.info('Diagnostic %s (before BN): mAP %.4f%%, Rank-1 %.4f%%', name,
                    100*values['mAP'], 100*values['rank1'])
    analysis_cfg = cfg.clone()
    analysis_cfg.defrost()
    epoch = model.checkpoint_epoch
    analysis_cfg.OUTPUT_DIR = str(Path(cfg.OUTPUT_DIR) / 'checkpoint_analysis' / f'epoch_{epoch:03d}')
    analysis_cfg.freeze()
    output = Path(analysis_cfg.OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    record_evaluation(output, epoch, metrics, diagnostics)
    record_weights(model, epoch, output, logger)
    if cfg.TEST.MLDC_VISUALIZE:
        save_visualizations(analysis_cfg, model, val_loader, num_query, epoch, logger)
    return metrics
