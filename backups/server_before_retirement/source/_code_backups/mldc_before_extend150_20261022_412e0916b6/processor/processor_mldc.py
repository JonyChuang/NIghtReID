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
from ..information_preservation import MultiLayerInformationPreservation
from ..shared_cnl import create_shared_memory
from ..cnl_schedule import auxiliary_weight_for_epoch, auxiliary_loss_for_step
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
                   'note': 'Same checkpoint, before-BN diagnostics; not independently trained ablations. '
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


def save_checkpoint(model, cfg, epoch, path, metrics=None, memory=None, dataset_fingerprint=None):
    payload = {'state_dict': model.state_dict(), 'epoch': epoch,
                'mldc_layers': list(model.mldc.selected_layers),
                'train_feature': getattr(model, 'train_feature', 'mldc'),
                'readout': model.readout, 'temperature': model.mldc.temperature,
                'config': cfg.dump(), 'metrics': metrics,
                'training_dataset_fingerprint': dataset_fingerprint}
    if cfg.SOLVER.AUXILIARY_LOSS == 'shared_cnl':
        payload.update(variant='mldc_multilayer_mined_retrieval_cnl_20261021',
            shared_cnl_spec=dict(layers=[9,10,11,12],matching='same_layer',neighbors='union',
                feature_space='shared_bn',contrast_space='fused_bn_after',
                memory_schema='image_index_fused_and_four_layers_v2',identity_labels_used=False,
                gamma=cfg.SOLVER.CNL_GAMMA,temperature=cfg.SOLVER.CNL_TEMPERATURE,
                momentum=cfg.SOLVER.CNL_MOMENTUM,cross_camera=cfg.SOLVER.CNL_CROSS_CAMERA),
            shared_cnl_memory=memory.state_dict() if memory is not None else None)
        if cfg.SOLVER.CNL_SELECTION == 'fused_consensus':
            payload['variant'] = 'mldc_fused_consensus_cnl_20261022'
            payload['shared_cnl_spec'].update(neighbors='fused_and_layer_consensus',
                selection=cfg.SOLVER.CNL_SELECTION,min_support=cfg.SOLVER.CNL_MIN_SUPPORT,
                uncertain_gamma=cfg.SOLVER.CNL_UNCERTAIN_GAMMA,
                uncertain_denominator_excluded=True,positive_maximum_required=True)
        payload['shared_cnl_spec'].update(
            warmup_epochs=getattr(cfg.SOLVER, 'CNL_WARMUP_EPOCHS', 0),
            weight_after_warmup=float(cfg.SOLVER.CNL_WEIGHT),
            effective_weight_at_save=auxiliary_weight_for_epoch(cfg, epoch),
            memory_during_warmup='update_after_successful_optimizer_steps')
    else:
        payload.update(variant=('mldc_information_preservation_bn_ratio_20261021'
                            if cfg.SOLVER.PRESERVE_REFERENCE_SPACE == 'shared_bn' and cfg.SOLVER.PRESERVE_TARGET_MODE == 'ratio'
                            else 'mldc_information_preservation_20261020'),
                preservation_reference_space=cfg.SOLVER.PRESERVE_REFERENCE_SPACE,
                preservation_target_mode=cfg.SOLVER.PRESERVE_TARGET_MODE,
                preservation_memory=memory.state_dict() if memory is not None else None)
    torch.save(payload, path)


def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn, num_query, dataset):
    device = torch.device('cuda:0')
    model.to(device)
    logger = logging.getLogger('transreid.train')
    output = Path(cfg.OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    shared_cnl = cfg.SOLVER.AUXILIARY_LOSS == 'shared_cnl'
    auxiliary_weight = cfg.SOLVER.CNL_WEIGHT if shared_cnl else cfg.SOLVER.PRESERVE_WEIGHT
    memory = (create_shared_memory(cfg, model, dataset, device) if shared_cnl else
        MultiLayerInformationPreservation([int(r[1]) for r in dataset.entries],
        dim=model.base.in_planes, num_layers=len(model.mldc.selected_layers),
        min_margin=cfg.SOLVER.PRESERVE_MIN_MARGIN, max_margin=cfg.SOLVER.PRESERVE_MAX_MARGIN,
        momentum=cfg.SOLVER.PRESERVE_MOMENTUM,
        hard_negatives=cfg.SOLVER.PRESERVE_HARD_NEGATIVES,
        target_mode=cfg.SOLVER.PRESERVE_TARGET_MODE,
        retain_ratio=cfg.SOLVER.PRESERVE_RETAIN_RATIO).to(device)
        if auxiliary_weight > 0 else None)
    scaler = amp.GradScaler()
    best_map = -1.0
    record_weights(model, 0, output, logger)
    if cfg.TEST.MLDC_VISUALIZE:
        save_visualizations(cfg, model, val_loader, num_query, 0, logger)
    for epoch in range(1, cfg.SOLVER.MAX_EPOCHS + 1):
        epoch_auxiliary_weight = auxiliary_weight_for_epoch(cfg, epoch)
        if shared_cnl:
            logger.info('Epoch[%d] CNL effective weight %.3f; warmup epochs %d; memory updates remain enabled',
                        epoch, epoch_auxiliary_weight, cfg.SOLVER.CNL_WARMUP_EPOCHS)
        start = time.monotonic()
        model.train()
        total_loss = total_correct = total_identity = total_triplet = samples = 0
        total_preservation = 0.
        batches = updates = 0
        preservation_totals = {}
        for iteration, (images, targets, camera, view, indices) in enumerate(train_loader, 1):
            images, targets = images.to(device), targets.to(device)
            camera, view, indices = camera.to(device), view.to(device), indices.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():
                scores, features, unit, layers = model(images, targets, cam_label=camera, view_label=view,
                    return_preservation_features=not shared_cnl, return_cnl_features=shared_cnl)
                original_loss = loss_fn(scores, features, targets, camera)
                preservation, stats = auxiliary_loss_for_step(memory, epoch_auxiliary_weight,
                    unit, layers, indices, camera if shared_cnl else targets)
                loss = original_loss + epoch_auxiliary_weight * preservation
            if not torch.isfinite(loss).item():
                raise FloatingPointError('Non-finite MLDC training objective')
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            applied = scaler.get_scale() >= scale_before
            if applied and memory is not None:
                if shared_cnl:
                    memory.update(unit, layers, indices, camera)
                else:
                    memory.update(unit, layers, indices, targets)
            batches += 1
            updates += int(applied)
            for key, value in stats.items():
                preservation_totals[key] = preservation_totals.get(key, 0) + value
            size = len(targets)
            samples += size
            total_loss += loss.item() * size
            total_preservation += float(preservation.detach()) * size
            total_correct += int((scores.argmax(1) == targets).sum().item())
            total_identity += loss_fn.last_terms['identity'].item() * size
            total_triplet += loss_fn.last_terms['triplet'].item() * size
            if iteration % cfg.SOLVER.LOG_PERIOD == 0:
                logger.info('Epoch[%d] Iter[%d/%d] Loss: %.4f, Acc: %.4f, ID: %.4f, Tri: %.4f, Base Lr: %.2e',
                            epoch, iteration, len(train_loader), total_loss / samples,
                            total_correct / samples, total_identity / samples,
                            total_triplet / samples, optimizer.param_groups[0]['lr'])
            if iteration == 1 or iteration % cfg.SOLVER.LOG_PERIOD == 0:
                logger.info('%s %.6f, successful updates %d/%d, statistics %s',
                            'Shared CNL' if shared_cnl else 'Preservation',
                            total_preservation / samples, updates, batches, stats)
        row = dict(epoch=epoch, global_id=total_identity / samples, global_triplet=total_triplet / samples,
            preservation=total_preservation / samples, loss=total_loss / samples, global_accuracy=total_correct / samples,
            batches=batches, optimizer_updates=updates, amp_skipped_steps=batches-updates,
            seconds=time.monotonic()-start,
            preservation_active_fraction=preservation_totals.get('active_anchors', 0)/max(preservation_totals.get('anchors', 0), 1),
            preservation_reliable_fraction=preservation_totals.get('reliable_triplets', 0)/max(preservation_totals.get('candidate_triplets', 0), 1),
            preservation_violated_fraction=preservation_totals.get('violated_triplets', 0)/max(preservation_totals.get('reliable_triplets', 0), 1),
            preservation_target_margin=preservation_totals.get('target_margin_sum', 0)/max(preservation_totals.get('reliable_triplets', 0), 1),
            preservation_reference_margin=preservation_totals.get('reference_margin_sum', 0)/max(preservation_totals.get('reliable_triplets', 0), 1),
            preservation_reference_correct_fusion_wrong_fraction=preservation_totals.get('reference_correct_fusion_wrong_triplets', 0)/max(preservation_totals.get('reliable_triplets', 0), 1),
            preservation_fused_margin=preservation_totals.get('fused_margin_sum', 0)/max(preservation_totals.get('reliable_triplets', 0), 1),
            memory_images=int(memory.valid.sum()) if memory is not None else 0)
        for layer, name in enumerate(model.mldc.selected_layers):
            row[f'preservation_reference_layer_{name}'] = preservation_totals.get(f'reference_layer_{layer}', 0)
        if shared_cnl:
            for key in list(row):
                if key.startswith('preservation'):
                    del row[key]
            anchors = max(preservation_totals.get('anchors', 0), 1)
            selected = max(preservation_totals.get('selected_positives', 0), 1)
            row.update(cnl=total_preservation / samples,
                cnl_effective_weight=epoch_auxiliary_weight,
                cnl_weighted=epoch_auxiliary_weight * total_preservation / samples,
                cnl_warmup=int(epoch <= cfg.SOLVER.CNL_WARMUP_EPOCHS),
                cnl_active_fraction=preservation_totals.get('active_anchors', 0)/anchors,
                cnl_empty_positive_fraction=preservation_totals.get('empty_positive_anchors', 0)/anchors,
                cnl_no_negative_fraction=preservation_totals.get('no_negative_anchors', 0)/anchors,
                cnl_mean_positives=preservation_totals.get('selected_positives', 0)/anchors,
                cnl_mean_out_of_batch_candidates=preservation_totals.get('out_of_batch_candidates', 0)/max(batches, 1),
                cnl_out_of_batch_positive_fraction=preservation_totals.get('out_of_batch_positives', 0)/selected,
                cnl_memory_coverage=float(memory.valid.float().mean()) if memory is not None else 0.,
                cnl_positive_fraction=preservation_totals.get('selected_positives', 0)/max(preservation_totals.get('eligible_pairs', 0), 1))
            if cfg.SOLVER.CNL_SELECTION == 'fused_consensus':
                eligible = max(preservation_totals.get('eligible_pairs', 0), 1)
                row.update(
                    cnl_mean_uncertain=preservation_totals.get('uncertain_pairs', 0)/anchors,
                    cnl_mean_negatives=preservation_totals.get('negative_pairs', 0)/anchors,
                    cnl_uncertain_fraction=preservation_totals.get('uncertain_pairs', 0)/eligible,
                    cnl_negative_fraction=preservation_totals.get('negative_pairs', 0)/eligible,
                    cnl_denominator_fraction=preservation_totals.get('denominator_pairs', 0)/eligible,
                    cnl_mean_union_candidates=preservation_totals.get('union_candidates', 0)/anchors,
                    cnl_mean_fused_confirmed_candidates=preservation_totals.get('fused_confirmed_candidates', 0)/anchors)
                for count in range(1,5):
                    row[f'cnl_candidate_support_{count}_fraction'] = preservation_totals.get(
                        f'candidate_support_{count}', 0)/max(preservation_totals.get('union_candidates', 0), 1)
            for count in range(1, 5):
                row[f'cnl_support_{count}_fraction'] = preservation_totals.get(f'support_{count}', 0)/selected
            for name in model.mldc.selected_layers:
                row[f'cnl_discovered_layer_{name}'] = preservation_totals.get(f'discovered_layer_{name}', 0)
        with (output / 'training.csv').open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if handle.tell() == 0: writer.writeheader()
            writer.writerow(row)
        scheduler.step(epoch)
        logger.info('Epoch %d done. Time: %.3fs', epoch, time.monotonic() - start)
        record_weights(model, epoch, output, logger)
        metrics = None
        if epoch % cfg.SOLVER.EVAL_PERIOD == 0:
            metrics, diagnostics = evaluate(cfg, model, val_loader, num_query, device, include_diagnostics=True)
            logger.info('Validation Results - Epoch: %d', epoch)
            for name, value in metrics.items():
                logger.info('%s: %.4f%%', name, value * 100)
            record_evaluation(output, epoch, metrics, diagnostics)
            for name, values in diagnostics['readouts'].items():
                logger.info('Diagnostic %s (before BN): mAP %.4f%%, Rank-1 %.4f%%', name,
                            100*values['mAP'], 100*values['rank1'])
            logger.info('Raw readout/last-CLS mean cosine: %.6f', diagnostics['raw_readout_last_cls_cosine_mean'])
            if metrics['mAP'] > best_map:
                best_map = metrics['mAP']
                save_checkpoint(model, cfg, epoch, output / 'transformer_best.pth', metrics, memory=memory, dataset_fingerprint=dataset.fingerprint)
                (output / 'best_metrics.json').write_text(json.dumps({'epoch': epoch, **metrics}, indent=2), encoding='utf-8')
                logger.info('Saved best mAP checkpoint at epoch %d', epoch)
        if epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS:
            save_checkpoint(model, cfg, epoch, output / f'transformer_{epoch}.pth', metrics, memory=memory, dataset_fingerprint=dataset.fingerprint)
        if cfg.TEST.MLDC_VISUALIZE and (epoch % cfg.TEST.MLDC_VIS_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS):
            save_visualizations(cfg, model, val_loader, num_query, epoch, logger)


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
