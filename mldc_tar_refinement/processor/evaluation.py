"""Original ReID evaluator, applied once to the main retrieval descriptor."""
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from utils.metrics import R1_mAP_eval


@torch.no_grad()
def evaluate(cfg, model, loader, num_query, device):
    model.eval()
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
                            reranking=cfg.TEST.RE_RANKING)
    evaluator.reset()
    for images, pids, camids, camera, view, _ in loader:
        feature = model(images.to(device), cam_label=camera.to(device), view_label=view.to(device))
        evaluator.update((feature, pids, camids))
    cmc, mean_ap, *_ = evaluator.compute()
    return {'mAP': float(mean_ap), 'rank1': float(cmc[0]),
            'rank5': float(cmc[4]), 'rank10': float(cmc[9])}


def diagnostic_features(parts):
    return {'main': parts['retrieval'], 'raw_residual': parts['raw_residual'],
            'raw_readout': parts['raw_readout'], 'mldc_after_bn': parts['mldc_after_bn'],
            'raw_tar': parts['raw_tar'], 'tar_after_bn': parts['tar_after_bn'],
            'raw_last_cls': parts['raw_last_cls'], 'raw_uniform_cls': parts['layers'].mean(0),
            'raw_plain_residual': parts['raw_readout'] + parts['raw_last_cls']}


@torch.no_grad()
def evaluate_diagnostics(cfg, model, loader, num_query, device, logger):
    """One extraction pass, then sequential original-evaluator distance matrices.

    Only the standalone testing entrypoint calls this. Training keeps evaluate().
    Each BN-after feature uses its separately supervised neck. The plain
    residual is a same-checkpoint diagnostic, not an independently trained control.
    """
    model.eval()
    start = time.monotonic()
    cached, labels = {}, []
    cosine_sum, sample_count = 0., 0
    logger.info('Retrieval diagnostics: extracting all requested features in one pass')
    for images, pids, camids, camera, view, _ in loader:
        parts = model(images.to(device), cam_label=camera.to(device),
                      view_label=view.to(device), return_components=True)
        features = diagnostic_features(parts)
        for name, feature in features.items():
            cached.setdefault(name, []).append(feature.detach().float().cpu())
        labels.append((pids, camids))
        cosine_sum += F.cosine_similarity(parts['raw_readout'].float(),
                                          parts['raw_last_cls'].float(), dim=1).sum().item()
        sample_count += len(images)
    logger.info('Feature extraction completed: %d images, %d cases, %.1fs',
                sample_count, len(cached), time.monotonic()-start)
    descriptions = {'main': 'Final residual (after its own BN)'}
    results, rows = {}, []
    for name, chunks in cached.items():
        case_start = time.monotonic()
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
                                reranking=cfg.TEST.RE_RANKING)
        evaluator.reset()
        for feature, (pids, camids) in zip(chunks, labels):
            evaluator.update((feature, pids, camids))
        computed = evaluator.compute()
        cmc, mean_ap = computed[:2]
        metrics = {'mAP': float(mean_ap), 'rank1': float(cmc[0]),
                   'rank5': float(cmc[4]), 'rank10': float(cmc[9])}
        results[name] = metrics
        rows.append({'epoch': model.checkpoint_epoch, 'feature': name,
                     'dimension': int(chunks[0].shape[1]), **metrics})
        logger.info('%s%s: mAP %.4f%%, Rank-1 %.4f%%, Rank-5 %.4f%%, Rank-10 %.4f%% (%.1fs)',
                    '' if name == 'main' else 'Diagnostic ', descriptions.get(name, name),
                    100*metrics['mAP'], 100*metrics['rank1'], 100*metrics['rank5'],
                    100*metrics['rank10'], time.monotonic()-case_start)
        # Release each distance matrix before allocating the next one.
        del computed, evaluator
    mean_cosine = cosine_sum/sample_count
    logger.info('Raw readout/last-CLS mean cosine: %.6f', mean_cosine)
    report = {'epoch': model.checkpoint_epoch, 'checkpoint': cfg.TEST.WEIGHT,
              'cases': results, 'raw_readout_last_cls_mean_cosine': mean_cosine,
              'normalization': cfg.TEST.FEAT_NORM, 'reranking': cfg.TEST.RE_RANKING,
              'note': 'Same-checkpoint readout diagnostics, not independently trained baselines.'}
    output = Path(cfg.OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    (output/'retrieval_diagnostics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (output/'retrieval_diagnostics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info('Retrieval diagnostics completed in %.1fs; saved to %s', time.monotonic()-start, output)
    return report


def append_csv(path, rows):
    if not rows:
        return
    with Path(path).open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)


def record_evaluation(output, epoch, metrics):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    append_csv(output/'evaluation.csv', [{'epoch': epoch, **metrics}])
