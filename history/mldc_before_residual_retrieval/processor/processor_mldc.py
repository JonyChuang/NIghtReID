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
    for name, evaluator in evaluators.items():
        cmc, mean_ap, *_ = evaluator.compute()
        results[name] = {'mAP': float(mean_ap), 'rank1': float(cmc[0]),
                         'rank5': float(cmc[4]), 'rank10': float(cmc[9])}
    if not include_diagnostics:
        return results['retrieval']
    weights = model.readout_weights().detach().double().cpu()
    weighted_norms = weights * norm_sum / count
    diagnostics = {'readouts': {name: results[name] for name in names[1:]},
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


def save_checkpoint(model, cfg, epoch, path, metrics=None):
    torch.save({'state_dict': model.state_dict(), 'epoch': epoch,
                'readout': model.readout, 'temperature': model.mldc.temperature,
                'config': cfg.dump(), 'metrics': metrics}, path)


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
        model.train()
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
                logger.info('Epoch[%d] Iter[%d/%d] Loss: %.4f, Acc: %.4f, ID: %.4f, Tri: %.4f, Base Lr: %.2e',
                            epoch, iteration, len(train_loader), total_loss / samples,
                            total_correct / samples, total_identity / samples,
                            total_triplet / samples, optimizer.param_groups[0]['lr'])
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
                save_checkpoint(model, cfg, epoch, output / 'transformer_best.pth', metrics)
                (output / 'best_metrics.json').write_text(json.dumps({'epoch': epoch, **metrics}, indent=2), encoding='utf-8')
                logger.info('Saved best mAP checkpoint at epoch %d', epoch)
        if epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS:
            save_checkpoint(model, cfg, epoch, output / f'transformer_{epoch}.pth', metrics)
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
