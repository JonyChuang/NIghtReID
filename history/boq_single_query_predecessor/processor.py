import csv
import gc
import json
import logging
import math
import time
from pathlib import Path
import numpy as np
import torch
from utils.metrics import R1_mAP_eval
from .loss import objectives


def append_csv(path, record):
    with Path(path).open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(record)


@torch.no_grad()
def evaluate(cfg, model, loader, num_query, device):
    model.eval()
    names = ('fused', 'global_same_final_bn', 'local_bn')
    evaluators = {name: R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=False) for name in names}
    for evaluator in evaluators.values():
        evaluator.reset()
    entropy_head = entropy_averaged = delta_ratio = bn_delta_ratio = 0.0
    count = 0
    for images, pids, camids, camera, view, _ in loader:
        data = model(images.to(device), cam_label=camera.to(device), view_label=view.to(device), diagnostics=True)
        for name in names:
            feature = data['output'] if name == 'fused' else data[name]
            evaluators[name].update((feature, pids, camids))
        a = data['attention'].double()  # Separate per-head and averaged-head statistics.
        h = -(a * a.clamp_min(1e-30).log()).sum(-1) / math.log(a.shape[-1])
        average = a.mean(1)
        entropy_head += h.mean((1,2)).sum().item()
        entropy_averaged += (-(average * average.clamp_min(1e-30).log()).sum(-1) / math.log(a.shape[-1])).mean(1).sum().item()
        delta_ratio += (data['delta'].float().norm(dim=-1) / data['global_feature'].float().norm(dim=-1).clamp_min(1e-6)).sum().item()
        bn_delta_ratio += ((data['output'] - data['global_same_final_bn']).float().norm(dim=-1) / data['global_same_final_bn'].float().norm(dim=-1).clamp_min(1e-6)).sum().item()
        count += len(images)
    results, rank1 = {}, {}
    for name in names:
        cmc, mean_ap, distance, pids, cams, *_ = evaluators[name].compute()
        results[name] = dict(mAP=float(mean_ap), rank1=float(cmc[0]), rank5=float(cmc[4]), rank10=float(cmc[9]))
        pid, cam = np.asarray(pids), np.asarray(cams)
        flags = np.full(num_query, -1, dtype=np.int8)
        for i in range(num_query):
            order = np.argsort(distance[i])
            keep = ~((pid[num_query:][order] == pid[i]) & (cam[num_query:][order] == cam[i]))
            relevant = pid[num_query:][order[keep]] == pid[i]
            if relevant.any():
                flags[i] = int(relevant[0])
        rank1[name] = flags
        del distance
        evaluators[name].reset()
        gc.collect()
    fused, global_only = rank1['fused'], rank1['global_same_final_bn']
    diagnostic = {'readouts': results, 'mean_per_head_attention_entropy': entropy_head/count,
                  'mean_averaged_head_attention_entropy': entropy_averaged/count,
                  'mean_delta_global_norm_ratio': delta_ratio/count,
                  'mean_bn_delta_global_norm_ratio': bn_delta_ratio/count,
                  'rank1_fixed_by_local': int(((fused == 1) & (global_only == 0)).sum()),
                  'rank1_broken_by_local': int(((fused == 0) & (global_only == 1)).sum()),
                  'global_readout_note': 'Local removal using the final fused BN; not an independently trained global baseline'}
    return results['fused'], diagnostic


def save_checkpoint(model, cfg, epoch, path, metrics=None):
    torch.save({'experiment': 'mldc_joint_local_v1_20261016', 'state_dict': model.state_dict(),
                'config': cfg.dump(), 'epoch': epoch, 'architecture': model.architecture(), 'metrics': metrics}, path)


def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, num_query):
    device = torch.device('cuda:0')
    model.to(device)
    logger = logging.getLogger('transreid.train')
    output = Path(cfg.OUTPUT_DIR)
    scaler = torch.amp.GradScaler('cuda')
    best_map = -1.0
    for epoch in range(1, cfg.SOLVER.MAX_EPOCHS+1):
        model.train()
        assert all(block.training for block in model.global_model.base.blocks)
        assert model.global_model.bottleneck.training
        start = time.monotonic()
        totals = dict(loss=0.0, identity=0.0, triplet=0.0, local_identity=0.0, local_triplet=0.0, final_accuracy=0.0, local_accuracy=0.0)
        count = 0
        for iteration, (images, targets, camera, view) in enumerate(train_loader, 1):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                data = model(images, targets, cam_label=camera.to(device), view_label=view.to(device))
                loss, terms = objectives(data, targets, cfg.SOLVER.LOCAL_LOSS_WEIGHT)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if iteration == 1 and (epoch == 1 or epoch % cfg.SOLVER.EVAL_PERIOD == 0):
                prefixes = ('global_model.base.blocks.', 'global_model.mldc.', 'reader.', 'global_model.bottleneck.', 'global_model.classifier.', 'local_classifier.')
                norms = {prefix: math.sqrt(sum(float(p.grad.detach().float().square().sum()) for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None)) for prefix in prefixes}
                logger.info('Epoch %d total-objective gradient norms: %s', epoch, norms)
            scaler.step(optimizer)
            scaler.update()
            size = len(targets)
            count += size
            totals['loss'] += loss.item()*size
            for name, value in terms.items():
                totals[name] += value.item()*size
            totals['final_accuracy'] += (data['scores'].argmax(-1) == targets).sum().item()
            totals['local_accuracy'] += (data['local_scores'].argmax(-1) == targets).sum().item()
            if iteration % cfg.SOLVER.LOG_PERIOD == 0:
                logger.info('Epoch[%d] Iter[%d/%d] loss=%.5f final_acc=%.4f local_acc=%.4f', epoch, iteration, len(train_loader), totals['loss']/count, totals['final_accuracy']/count, totals['local_accuracy']/count)
        scheduler.step(epoch)  # Same timing as the matched deep-four baseline.
        append_csv(output/'training.csv', {'epoch':epoch, **{k:v/count for k,v in totals.items()}, 'seconds':time.monotonic()-start})
        append_csv(output/'layer_weights.csv', {'epoch':epoch, **{'layer_%d'%(i+1):float(w) for i,w in enumerate(model.global_model.readout_weights().detach().cpu())}})
        metrics = None
        if epoch % cfg.SOLVER.EVAL_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS:
            metrics, diagnostics = evaluate(cfg, model, val_loader, num_query, device)
            append_csv(output/'evaluation.csv', {'epoch':epoch, **metrics})
            with (output/'evaluation_diagnostics.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'epoch':epoch, **diagnostics})+'\n')
            logger.info('Epoch %d metrics=%s diagnostics=%s', epoch, metrics, diagnostics)
            if metrics['mAP'] > best_map:
                best_map = metrics['mAP']
                save_checkpoint(model, cfg, epoch, output/'transformer_best.pth', metrics)
        if epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS:
            save_checkpoint(model, cfg, epoch, output/('transformer_%d.pth'%epoch), metrics)
