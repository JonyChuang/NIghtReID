"""Isolated stage-1 training and evaluation for patch-compensated MLDC."""
import csv
import json
import logging
import math
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
    names = ['retrieval'] + (['anchor_bn', 'local_bn', 'raw_readout', 'raw_last_cls', 'raw_uniform_cls']
                             if include_diagnostics else [])
    evaluators = {name: R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
                                  reranking=cfg.TEST.RE_RANKING) for name in names}
    for evaluator in evaluators.values():
        evaluator.reset()
    gate_sum = gate_square_sum = entropy_sum = 0.0
    gate_min, gate_max = float("inf"), float("-inf")
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
                        'local_bn': data['local_bn'], 'raw_readout': raw, 'raw_last_cls': layers[-1],
                        'raw_uniform_cls': selected.mean(0)}
            gates = data['gate'].double()
            gate_sum += gates.sum().item()
            gate_square_sum += gates.square().sum().item()
            gate_min = min(gate_min, gates.min().item())
            gate_max = max(gate_max, gates.max().item())
            probability = data['patch_attention'].double().mean(1).flatten(1)
            entropy_sum += (-(probability * probability.clamp_min(1e-30).log()).sum(1) /
                            math.log(probability.shape[1])).sum().item()
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
                   'stage': model.stage, 'training_epoch': int(model._training_epoch.item()),
                   'mean_gate': gate_sum / count,
                   'std_gate': math.sqrt(max(0.0, gate_square_sum / count - (gate_sum / count)**2)),
                   'min_gate': gate_min, 'max_gate': gate_max,
                   'mean_attention_entropy': entropy_sum / count,
                   'mean_patch_layer_mass': (layer_sum / count).tolist(),
                   'mean_delta_anchor_norm_ratio': ratio_sum / count,
                   'note': 'anchor_bn uses the frozen original BN; local_bn uses the auxiliary local BN. Other readouts '
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
    if int(model._training_epoch.item()) != epoch:
        raise ValueError('Save the checkpoint at its actual phase/epoch')
    torch.save({'state_dict': model.state_dict(), 'epoch': epoch,
                'mldc_layers': list(model.mldc.selected_layers),
                'train_feature': model.train_feature, 'readout': model.readout,
                'temperature': model.mldc.temperature, 'config': cfg.dump(),
                'metrics': metrics, **model.checkpoint_metadata()}, path)


def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn, num_query):
    if model.checkpoint_epoch != 0:
        raise ValueError('This entry starts a fresh two-stage run from the baseline; optimizer resume is not implemented')
    device = torch.device('cuda:0')
    model.to(device)
    logger = logging.getLogger('transreid.train')
    output = Path(cfg.OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    scaler = amp.GradScaler()
    best_map = -1.0
    model.set_epoch(0)
    record_weights(model, 0, output, logger)
    if cfg.TEST.MLDC_VISUALIZE:
        save_visualizations(cfg, model, val_loader, num_query, 0, logger)
    for epoch in range(1, cfg.SOLVER.MAX_EPOCHS + 1):
        start = time.monotonic()
        model.set_epoch(epoch)
        model.train()
        sums = dict(loss=0.0, identity=0.0, triplet=0.0, local_identity=0.0, local_triplet=0.0,
                    correct=0, local_correct=0, gate=0.0)
        samples = 0
        for iteration, (images, targets, camera, view) in enumerate(train_loader, 1):
            images, targets = images.to(device), targets.to(device)
            camera, view = camera.to(device), view.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():
                data = model(images, targets, cam_label=camera, view_label=view)
                loss = loss_fn(data, targets, camera)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            size = len(targets)
            samples += size
            sums['loss'] += loss.item() * size
            sums['correct'] += int((data['scores'].argmax(1) == targets).sum().item())
            sums['local_correct'] += int((data['local_scores'].argmax(1) == targets).sum().item())
            sums['gate'] += data['gate'].detach().sum().item()
            for key in ('identity', 'triplet', 'local_identity', 'local_triplet'):
                sums[key] += loss_fn.last_terms[key].item() * size
            if iteration % cfg.SOLVER.LOG_PERIOD == 0:
                logger.info('Epoch[%d] Stage[%s] Iter[%d/%d] Loss %.4f GlobalAcc %.4f LocalAcc %.4f '
                            'ID %.4f Tri %.4f LocalID %.4f LocalTri %.4f Gate %.5f PatchLR %.2e',
                            epoch, model.stage, iteration, len(train_loader), sums['loss']/samples,
                            sums['correct']/samples, sums['local_correct']/samples,
                            sums['identity']/samples, sums['triplet']/samples,
                            sums['local_identity']/samples, sums['local_triplet']/samples,
                            sums['gate']/samples, optimizer.param_groups[0]['lr'])
        if samples == 0:
            raise ValueError('Empty training loader')
        with (output / 'training.csv').open('a', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            if stream.tell() == 0:
                writer.writerow(['epoch', 'stage', 'samples'] + list(sums))
            writer.writerow([epoch, model.stage, samples] + [sums[k]/samples for k in sums])
        scheduler.step(epoch)
        logger.info('Epoch %d stage=%s done. Time %.3fs', epoch, model.stage, time.monotonic()-start)
        record_weights(model, epoch, output, logger)
        boundary = epoch == model.warmup_epochs
        metrics = None
        if boundary or epoch % cfg.SOLVER.EVAL_PERIOD == 0:
            metrics, diagnostics = evaluate(cfg, model, val_loader, num_query, device, True)
            record_evaluation(output, epoch, metrics, diagnostics)
            logger.info('Epoch[%d] stage=%s retrieval=%s anchor_bn=%s local_bn=%s '
                        'gate=%.6f attention_entropy=%.6f delta/anchor=%.6f',
                        epoch, model.stage, metrics, diagnostics['readouts']['anchor_bn'],
                        diagnostics['readouts']['local_bn'], diagnostics['mean_gate'],
                        diagnostics['mean_attention_entropy'], diagnostics['mean_delta_anchor_norm_ratio'])
            if metrics['mAP'] > best_map:
                best_map = metrics['mAP']
                save_checkpoint(model, cfg, epoch, output/'transformer_best.pth', metrics)
                (output/'best_metrics.json').write_text(json.dumps({'epoch': epoch, 'stage': model.stage, **metrics}, indent=2))
        if boundary:
            save_checkpoint(model, cfg, epoch, output/'transformer_local_warmup.pth', metrics)
        if epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS:
            save_checkpoint(model, cfg, epoch, output/'transformer_{}.pth'.format(epoch), metrics)
        if cfg.TEST.MLDC_VISUALIZE and (boundary or epoch % cfg.TEST.MLDC_VIS_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS):
            save_visualizations(cfg, model, val_loader, num_query, epoch, logger)


def do_inference(cfg, model, val_loader, num_query):
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    metrics, diagnostics = evaluate(cfg, model, val_loader, num_query, device, True)
    logger = logging.getLogger('transreid.test')
    logger.info('Stage=%s Retrieval=%s; frozen anchor=%s; local=%s', model.stage, metrics,
                diagnostics['readouts']['anchor_bn'], diagnostics['readouts']['local_bn'])
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
