"""Isolated stage-1 training and evaluation for patch-compensated MLDC."""
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


def record_weights(model, epoch, output, logger):
    weights = model.readout_weights().detach().float().cpu().tolist()
    path = Path(output) / 'layer_weights.csv'
    with path.open('w' if epoch == 0 else 'a', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(['epoch'] + ['layer_{}'.format(i) for i in range(1, 13)])
        writer.writerow([epoch] + weights)
    logger.info('Epoch[%d] Frozen CLS weights: %s', epoch, weights)


@torch.no_grad()
def evaluate(cfg, model, loader, num_query, device, include_diagnostics=False):
    model.eval()
    names = ['retrieval'] + (['anchor_bn', 'raw_readout', 'raw_last_cls', 'raw_uniform_cls']
                             if include_diagnostics else [])
    evaluators = {name: R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
                                  reranking=cfg.TEST.RE_RANKING) for name in names}
    for evaluator in evaluators.values():
        evaluator.reset()
    null_sum = torch.zeros(model.patch_compensation.heads, dtype=torch.float64)
    layer_sum = torch.zeros(len(model.patch_compensation.layers), dtype=torch.float64)
    norm_sum = torch.zeros(12, dtype=torch.float64)
    ratio_sum = cosine_sum = 0.0
    count = 0
    for images, pids, camids, camera, view, _ in loader:
        if include_diagnostics:
            data = model(images.to(device), cam_label=camera.to(device),
                         view_label=view.to(device), return_patch_diagnostics=True)
            layers, raw, anchor = data['cls_features'], data['raw'], data['anchor']
            selected = layers.index_select(0, model.mldc.selected_indices)
            features = {'retrieval': data['output'], 'anchor_bn': model.bottleneck(anchor),
                        'raw_readout': raw, 'raw_last_cls': layers[-1],
                        'raw_uniform_cls': selected.mean(0)}
            null_sum += data['null_mass'].double().sum(0).cpu()
            layer_sum += data['layer_mass'].double().mean(1).sum(0).cpu()
            ratio_sum += (data['delta'].float().norm(dim=-1) /
                          anchor.float().norm(dim=-1).clamp_min(1e-12)).sum().item()
            norm_sum += layers.float().norm(dim=-1).double().sum(1).cpu()
            cosine_sum += F.cosine_similarity(raw.float(), layers[-1].float()).sum().item()
            count += len(images)
        else:
            features = {'retrieval': model(images.to(device), cam_label=camera.to(device),
                                            view_label=view.to(device))}
        for name, evaluator in evaluators.items():
            evaluator.update((features[name], pids, camids))
    results = {}
    for name, evaluator in evaluators.items():
        cmc, mean_ap, *_ = evaluator.compute()
        results[name] = {'mAP': float(mean_ap), 'rank1': float(cmc[0]),
                         'rank5': float(cmc[4]), 'rank10': float(cmc[9])}
    if not include_diagnostics:
        return results['retrieval']
    if count == 0:
        raise ValueError('Empty validation loader')
    diagnostics = {'retrieval_mode': 'compensated_mldc', 'training_feature': model.train_feature,
                   'readouts': {name: results[name] for name in names[1:]},
                   'layer_weights': model.readout_weights().detach().cpu().tolist(),
                   'mean_raw_layer_norms': (norm_sum / count).tolist(),
                   'raw_readout_last_cls_cosine_mean': cosine_sum / count,
                   'patch_layers': list(model.patch_compensation.layers),
                   'mean_null_mass_per_head': (null_sum / count).tolist(),
                   'mean_patch_layer_mass': (layer_sum / count).tolist(),
                   'mean_delta_anchor_norm_ratio': ratio_sum / count,
                   'note': 'anchor_bn uses the frozen original BN. Other diagnostic readouts '
                           'are before BN; uniform CLS includes only layers 9-12. '
                           'Attention allocation does not establish causal identity evidence.'}
    return results['retrieval'], diagnostics


def record_evaluation(output, epoch, metrics, diagnostics):
    output = Path(output)
    with (output / 'evaluation.csv').open('a', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(['epoch', 'mAP', 'rank1', 'rank5', 'rank10'])
        writer.writerow([epoch] + [metrics[k] for k in ('mAP', 'rank1', 'rank5', 'rank10')])
    with (output / 'diagnostics.csv').open('a', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(['epoch', 'feature', 'mAP', 'rank1', 'rank5', 'rank10'])
        for name, row in diagnostics['readouts'].items():
            writer.writerow([epoch, name] + [row[k] for k in ('mAP', 'rank1', 'rank5', 'rank10')])
    with (output / 'evaluation_diagnostics.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({'epoch': epoch, 'retrieval': metrics, **diagnostics}) + '\n')


def save_checkpoint(model, cfg, epoch, path, metrics=None):
    torch.save({'state_dict': model.state_dict(), 'epoch': epoch,
                'mldc_layers': list(model.mldc.selected_layers),
                'train_feature': model.train_feature, 'readout': model.readout,
                'temperature': model.mldc.temperature, 'config': cfg.dump(),
                'metrics': metrics, **model.checkpoint_metadata()}, path)


def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn, num_query):
    device = torch.device('cuda:0')
    model.to(device)
    logger = logging.getLogger('transreid.train')
    output = Path(cfg.OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    scaler = amp.GradScaler()
    best_map = -1.0
    record_weights(model, 0, output, logger)
    if cfg.TEST.MLDC_VISUALIZE:
        save_visualizations(cfg, model, val_loader, num_query, 0, logger)
    for epoch in range(1, cfg.SOLVER.MAX_EPOCHS + 1):
        start = time.monotonic()
        model.train()  # override keeps backbone, CLS fusion and BN in eval mode
        total_loss = total_correct = total_identity = total_triplet = samples = 0
        for iteration, (images, targets, camera, view) in enumerate(train_loader, 1):
            images, targets = images.to(device), targets.to(device)
            camera, view = camera.to(device), view.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():
                scores, features = model(images, targets, cam_label=camera, view_label=view)
                loss = loss_fn(scores, features, targets, camera)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            size = len(targets)
            samples += size
            total_loss += loss.item() * size
            total_correct += int((scores.argmax(1) == targets).sum().item())
            total_identity += loss_fn.last_terms['identity'].item() * size
            total_triplet += loss_fn.last_terms['triplet'].item() * size
            if iteration % cfg.SOLVER.LOG_PERIOD == 0:
                logger.info('Epoch[%d] Iter[%d/%d] Loss %.4f Acc %.4f ID %.4f Tri %.4f LR %.2e',
                            epoch, iteration, len(train_loader), total_loss / samples,
                            total_correct / samples, total_identity / samples,
                            total_triplet / samples, optimizer.param_groups[0]['lr'])
        scheduler.step(epoch)
        logger.info('Epoch %d done. Time %.3fs', epoch, time.monotonic() - start)
        record_weights(model, epoch, output, logger)
        metrics = None
        if epoch % cfg.SOLVER.EVAL_PERIOD == 0:
            metrics, diagnostics = evaluate(cfg, model, val_loader, num_query, device, True)
            record_evaluation(output, epoch, metrics, diagnostics)
            logger.info('Epoch[%d] retrieval=%s anchor_bn=%s null/head=%s delta/anchor=%.6f',
                        epoch, metrics, diagnostics['readouts']['anchor_bn'],
                        diagnostics['mean_null_mass_per_head'], diagnostics['mean_delta_anchor_norm_ratio'])
            if metrics['mAP'] > best_map:
                best_map = metrics['mAP']
                save_checkpoint(model, cfg, epoch, output / 'transformer_best.pth', metrics)
                (output / 'best_metrics.json').write_text(json.dumps({'epoch': epoch, **metrics}, indent=2))
        if epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS:
            save_checkpoint(model, cfg, epoch, output / 'transformer_{}.pth'.format(epoch), metrics)
        if cfg.TEST.MLDC_VISUALIZE and (epoch % cfg.TEST.MLDC_VIS_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS):
            save_visualizations(cfg, model, val_loader, num_query, epoch, logger)


def do_inference(cfg, model, val_loader, num_query):
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    metrics, diagnostics = evaluate(cfg, model, val_loader, num_query, device, True)
    logger = logging.getLogger('transreid.test')
    logger.info('Retrieval=%s; frozen anchor BN=%s', metrics, diagnostics['readouts']['anchor_bn'])
    analysis_cfg = cfg.clone()
    analysis_cfg.defrost()
    analysis_cfg.OUTPUT_DIR = str(Path(cfg.OUTPUT_DIR) / 'checkpoint_analysis' /
                                  'epoch_{:03d}'.format(model.checkpoint_epoch))
    analysis_cfg.freeze()
    output = Path(analysis_cfg.OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    record_evaluation(output, model.checkpoint_epoch, metrics, diagnostics)
    record_weights(model, model.checkpoint_epoch, output, logger)
    if cfg.TEST.MLDC_VISUALIZE:
        save_visualizations(analysis_cfg, model, val_loader, num_query, model.checkpoint_epoch, logger)
    return metrics
