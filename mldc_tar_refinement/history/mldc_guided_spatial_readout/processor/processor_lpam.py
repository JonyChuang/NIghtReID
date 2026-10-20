"""Joint training from epoch one, with full epoch-boundary resume."""
import csv
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .evaluation import evaluate, evaluate_diagnostics, record_evaluation, append_csv
from .visualization import save_visualizations
from ..model.make_model_lpam import VARIANT


def learning_rates(optimizer):
    return {group['name']: group['lr'] for group in optimizer.param_groups}


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def checkpoint_state(model, cfg, epoch, optimizer, scaler, best_map, metrics=None, scheduler=None):
    return {'state_dict': model.state_dict(), 'variant': VARIANT, 'architecture': model.architecture(),
            'epoch': epoch, 'phase': model.phase, 'training_mode': 'joint_from_start',
            'fft': False, 'config': cfg.dump(), 'fusion_type': 'adaptive_global_layer_reads',
            'fusion_initial_local_weight': model.initial_local_weight,
            'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'rng': rng_state(),
            'scheduler': scheduler.state_dict(), 'lr_policy': 'original_mldc_scheduler',
            'best_mAP': best_map, 'metrics': metrics,
            'primary_endpoint': int(cfg.SOLVER.MAX_EPOCHS)}


def atomic_save(state, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(state, temporary)
    os.replace(str(temporary), str(path))


def trim_history(output, epoch):
    # An interruption after logging but before checkpoint commit may leave extra rows.
    for name in ('evaluation.csv', 'diagnostics.csv', 'training.csv', 'layer_weights.csv'):
        path = Path(output)/name
        if not path.exists():
            continue
        with path.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, [r for r in reader if int(r['epoch']) <= epoch]
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    path = Path(output)/'evaluation_diagnostics.jsonl'
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows if row['epoch'] <= epoch), encoding='utf-8')


def record_weights(model, epoch, output, logger):
    weights = model.readout_weights().detach().cpu().tolist()
    append_csv(Path(output)/'layer_weights.csv', [{'epoch': epoch, **{'layer_{}'.format(i): w for i, w in enumerate(weights, 1)}}])
    logger.info('Epoch[%d] mldc layer weights: %s', epoch, ', '.join('{}:{:.5f}'.format(i, w) for i, w in enumerate(weights, 1)))


def evaluate_epoch(cfg, model, loader, num_query, epoch, device, logger, force_visual=False):
    start = time.monotonic()
    logger.info('Validation Results - Epoch: %d', epoch)
    metrics = evaluate(cfg, model, loader, num_query, device)
    record_evaluation(cfg.OUTPUT_DIR, epoch, metrics)
    for name, value in metrics.items():
        logger.info('%s: %.4f%%', name, 100*value)
    logger.info('Validation completed in %.1fs', time.monotonic()-start)
    if cfg.TEST.MLDC_VISUALIZE and (force_visual or epoch % cfg.TEST.MLDC_VIS_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS):
        save_visualizations(cfg, model, loader, num_query, epoch, logger)
    return metrics


def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn, num_query, resume=None):
    device = torch.device('cuda:0')
    model.to(device)
    logger = logging.getLogger('transreid.train')
    output = Path(cfg.OUTPUT_DIR)
    scaler = torch.cuda.amp.GradScaler()
    start_epoch, best_map = 1, -1.
    if resume is not None:
        if resume['config'] != cfg.dump():
            raise ValueError('Resume requires exactly the recorded experiment config')
        optimizer.load_state_dict(resume['optimizer'])
        scaler.load_state_dict(resume['scaler'])
        scheduler.load_state_dict(resume['scheduler'])
        start_epoch, best_map = int(resume['epoch'])+1, float(resume['best_mAP'])
        trim_history(output, start_epoch-1)
        restore_rng(resume['rng'])
        logger.info('Resuming after epoch %d, phase %s', start_epoch-1, resume['phase'])
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS+1):
        start = time.monotonic()
        model.train()
        rates = learning_rates(optimizer)
        logger.info('Epoch[%d] phase=%s learning_rates=%s', epoch, model.phase, rates)
        sums, samples, global_correct, read_correct = {}, 0, 0, 0
        fusion_weight_sum = torch.zeros(model.lpam.branches+1, device=device)
        for iteration, (images, target, camera, view) in enumerate(train_loader, 1):
            images, target = images.to(device), target.to(device)
            camera, view = camera.to(device), view.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                parts = model(images, target, cam_label=camera, view_label=view)
                loss = loss_fn(parts, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            size = len(target)
            samples += size
            for name, value in {'loss': loss.detach(), **loss_fn.last_terms}.items():
                sums[name] = sums.get(name, 0.)+value.item()*size
            global_correct += (parts['global_logits'].argmax(1) == target).sum().item()
            read_correct += (parts['read_logits'].argmax(1) == target).sum().item()
            fusion_weight_sum += parts['fusion_weights'].detach().sum(0)
            if iteration % cfg.SOLVER.LOG_PERIOD == 0:
                logger.info('Epoch[%d] Iter[%d/%d] phase=%s terms=%s read_acc=%.4f global_acc=%.4f',
                            epoch, iteration, len(train_loader), model.phase,
                            {k: round(v/samples, 5) for k, v in sums.items()}, read_correct/samples, global_correct/samples)
        if not samples:
            raise ValueError('Empty training loader')
        mean_fusion = (fusion_weight_sum/samples).cpu().tolist()
        logger.info('Epoch[%d] mean fusion weights %s=%s', epoch,
                    ['global']+['layer{}'.format(layer) for layer in model.patch_layers],
                    [round(v, 5) for v in mean_fusion])
        append_csv(output/'training.csv', [{'epoch': epoch, 'phase': model.phase,
                                           **{k: v/samples for k, v in sums.items()},
                                           'read_accuracy': read_correct/samples,
                                           'global_accuracy': global_correct/samples,
                                           **{name: mean_fusion[i] for i, name in enumerate(
                                               ['fusion_global_weight']+['fusion_layer_{}_weight'.format(layer)
                                                                       for layer in model.patch_layers])},
                                           **{k+'_lr': v for k, v in rates.items()}}])
        scheduler.step(epoch)
        record_weights(model, epoch, output, logger)
        metrics = None
        improved = False
        if epoch % cfg.SOLVER.EVAL_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS:
            metrics = evaluate_epoch(cfg, model, val_loader, num_query, epoch, device, logger)
            improved = metrics['mAP'] > best_map
            if improved:
                best_map = metrics['mAP']
        state = checkpoint_state(model, cfg, epoch, optimizer, scaler, best_map, metrics, scheduler=scheduler)
        atomic_save(state, output/'transformer_latest.pth')
        if epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS:
            atomic_save(state, output/'transformer_{}.pth'.format(epoch))
        if improved:
            atomic_save(state, output/'transformer_best.pth')
            (output/'best_metrics.json').write_text(json.dumps({'epoch': epoch, **metrics,
                'selection': 'Night600 test mAP selected; diagnostic only, final scheduled epoch is primary'}, indent=2), encoding='utf-8')
        logger.info('Epoch[%d] completed in %.1fs; latest checkpoint saved', epoch, time.monotonic()-start)
    logger.info('Training finished. Primary endpoint: transformer_%d.pth', cfg.SOLVER.MAX_EPOCHS)


def do_inference(cfg, model, val_loader, num_query, diagnostics=False, layer_diagnostics=False):
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    analysis_cfg = cfg.clone()
    analysis_cfg.defrost()
    analysis_cfg.OUTPUT_DIR = str(Path(cfg.OUTPUT_DIR)/'checkpoint_analysis'/'epoch_{:03d}'.format(model.checkpoint_epoch))
    analysis_cfg.freeze()
    if diagnostics or layer_diagnostics:
        logger = logging.getLogger('transreid.test')
        report = evaluate_diagnostics(analysis_cfg, model, val_loader, num_query, device,
                                      logger, include_layers=layer_diagnostics)
        if cfg.TEST.MLDC_VISUALIZE:
            save_visualizations(analysis_cfg, model, val_loader, num_query, model.checkpoint_epoch, logger)
        return report
    return evaluate_epoch(analysis_cfg, model, val_loader, num_query, model.checkpoint_epoch, device, logging.getLogger('transreid.test'), force_visual=True)
