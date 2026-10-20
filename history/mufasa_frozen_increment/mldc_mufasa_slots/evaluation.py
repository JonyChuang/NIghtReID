"""Original retrieval evaluator, with paired query-level diagnostics."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from utils.metrics import R1_mAP_eval


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def query_results(distance, pids, cams, num_query, paths):
    pids, cams = np.asarray(pids), np.asarray(cams)
    gallery_pids, gallery_cams = pids[num_query:], cams[num_query:]
    rows = []
    for index, distances in enumerate(distance):
        order = np.argsort(distances)
        keep = ~((gallery_pids[order] == pids[index]) & (gallery_cams[order] == cams[index]))
        order = order[keep]
        positive = gallery_pids[order] == pids[index]
        if not positive.any():
            continue
        precision = positive.cumsum() / np.arange(1, len(order)+1)
        first = int(np.flatnonzero(positive)[0])+1
        top = int(order[0])+num_query
        rows.append({'query_index': index, 'pid': int(pids[index]), 'camera': int(cams[index]),
                     'path': str(paths[index]), 'AP': float(precision[positive].mean()),
                     'rank1': int(first == 1), 'first_positive_rank': first,
                     'top1_gallery_index': top, 'top1_path': str(paths[top])})
    return rows


def diagnostic_values(parts):
    maps = parts['maps'].float()
    k = maps.shape[-1]
    occupancy = F.one_hot(maps.argmax(-1), k).float().mean(-2)
    slots = F.normalize(parts['layer_slots'].float(), dim=-1)
    pair_cosine = slots @ slots.transpose(-1, -2)
    offdiag = ~torch.eye(k, dtype=torch.bool, device=slots.device)
    g = F.normalize(parts['global_feature'].float(), dim=-1)
    r = F.normalize(parts['residual'].float(), dim=-1)
    cosine = (g*r).sum(-1)
    attention = parts['readout_attention'].float()
    result = {
        'global_residual_cosine': cosine,
        'residual_orthogonal_norm': (r-cosine[:, None]*g).norm(dim=-1),
        'global_final_cosine': (g*parts['feature']).sum(-1),
        'raw_residual_norm': parts['residual'].float().norm(dim=-1),
        'assignment_entropy': -(maps.clamp_min(1e-12).log()*maps).sum(-1).mean(dim=(1, 2)),
        'readout_entropy': -(attention.clamp_min(1e-12).log()*attention).sum(-1),
        'match_iou_before': parts['match_before'].mean(-1),
        'match_iou_after': parts['match_after'].mean(-1),
        'empty_slot_fraction': (occupancy == 0).float().mean(dim=(1, 2)),
    }
    if k > 1:
        result['within_layer_slot_cosine'] = pair_cosine[..., offdiag].mean(dim=(1, 2))
    for layer in range(maps.shape[1]):
        for slot in range(k):
            result[f'branch_{layer+1}_slot_{slot+1}_occupancy'] = occupancy[:, layer, slot]
    for slot in range(k):
        result[f'readout_slot_{slot+1}_weight'] = attention[:, slot]
    return result


@torch.no_grad()
def evaluate(cfg, model, loader, num_query, destination):
    model.eval()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    evaluators = {name: R1_mAP_eval(num_query, max_rank=50, feat_norm=True, reranking=False)
                  for name in ('baseline', 'final', 'residual_only')}
    for evaluator in evaluators.values():
        evaluator.reset()
    sums, count, paths = {}, 0, []
    residual_sum = torch.zeros(768, dtype=torch.float64)
    device = next(model.parameters()).device
    for images, pids, cams, camera, view, image_paths in loader:
        parts = model(images.to(device), cam_label=camera.to(device), view_label=view.to(device),
                      return_parts=True, reconstruct=False)
        for name, feature in [('baseline', parts['global_feature']), ('final', parts['feature']),
                              ('residual_only', parts['residual'])]:
            evaluators[name].update((feature, pids, cams))
        for name, values in diagnostic_values(parts).items():
            sums[name] = sums.get(name, 0.) + values.double().sum().item()
        residual_sum += F.normalize(parts['residual'].float(), dim=-1).double().sum(0).cpu()
        paths.extend(image_paths)
        count += len(images)
    metrics, queries = {}, {}
    for name, evaluator in evaluators.items():
        cmc, mean_ap, distance, pids, cams, _, _ = evaluator.compute()
        metrics[name] = {'mAP': float(mean_ap), 'rank1': float(cmc[0]),
                         'rank5': float(cmc[4]), 'rank10': float(cmc[9])}
        queries[name] = query_results(distance, pids, cams, num_query, paths)
        # Ensure the paired diagnostics implement the original protocol.
        assert abs(np.mean([r['AP'] for r in queries[name]]) - mean_ap) < 1e-10
        assert abs(np.mean([r['rank1'] for r in queries[name]]) - float(cmc[0])) < 1e-6
        write_csv(destination / f'queries_{name}.csv', queries[name])
        del distance
    baseline = {r['query_index']: r for r in queries['baseline']}
    changes = []
    for row in queries['final']:
        before = baseline[row['query_index']]
        changes.append({'query_index': row['query_index'], 'pid': row['pid'],
                        'baseline_AP': before['AP'], 'final_AP': row['AP'],
                        'delta_AP': row['AP']-before['AP'],
                        'baseline_rank1': before['rank1'], 'final_rank1': row['rank1'],
                        'fixed': int(row['rank1'] and not before['rank1']),
                        'broken': int(before['rank1'] and not row['rank1'])})
    write_csv(destination / 'query_changes.csv', changes)
    diagnostics = {key: value/count for key, value in sums.items()}
    diagnostics['normalized_residual_mean_norm'] = float((residual_sum/count).norm())
    # High mean norm signals an image-independent common residual direction.
    diagnostics.update(fixed_queries=sum(r['fixed'] for r in changes),
                       broken_queries=sum(r['broken'] for r in changes),
                       num_query=num_query, num_images=count)
    report = {'metrics': metrics, 'diagnostics': diagnostics}
    (destination / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def assert_baseline(metrics, expected, tolerance=1e-6):
    for name in ('mAP', 'rank1', 'rank5', 'rank10'):
        if abs(metrics[name]-expected[name]) > tolerance:
            raise RuntimeError('Frozen MLDC does not reproduce its checkpoint metrics: '
                               f'{name}={metrics[name]:.9f}, expected {expected[name]:.9f}. '
                               'Check data/preprocessing/environment before training.')
