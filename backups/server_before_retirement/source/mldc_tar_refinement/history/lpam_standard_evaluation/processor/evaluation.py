"""Same Market protocol, plus per-query evidence of complementarity."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def rank_features(features, pids, cameras, num_query, max_rank=50):
    features = F.normalize(features.float(), dim=1)
    similarities = features[:num_query] @ features[num_query:].T
    # Squared Euclidean distance of unit vectors, as in the original evaluator.
    distances = (2-2*similarities).numpy()
    orders = np.argsort(distances, axis=1)
    gp, gc = pids[num_query:], cameras[num_query:]
    rows, curves = [], []
    for i, order in enumerate(orders):
        order = order[~((gp[order] == pids[i]) & (gc[order] == cameras[i]))]
        matches = gp[order] == pids[i]
        valid = bool(matches.any())
        ap = float(((matches.cumsum()/np.arange(1, len(matches)+1))*matches).sum()/matches.sum()) if valid else None
        if valid:
            curve = np.minimum(1, matches.cumsum())[:max_rank]
            curve = np.pad(curve, (0, max_rank-len(curve)), mode='edge')
            curves.append(curve)
        rows.append({'query_index': i, 'valid': valid, 'AP': ap,
                     'top1_correct': bool(matches[0]) if valid else None,
                     'first_positive_rank': int(np.flatnonzero(matches)[0]+1) if valid else None,
                     'top10': (order[:10]+num_query).tolist()})
    if not curves:
        raise ValueError('No valid cross-camera positives in evaluation')
    cmc = np.asarray(curves, dtype=np.float32).mean(0)
    metrics = {'mAP': float(np.mean([r['AP'] for r in rows if r['valid']])),
               'rank1': float(cmc[0]), 'rank5': float(cmc[4]), 'rank10': float(cmc[9])}
    return metrics, rows


def complementarity(global_rows, joint_rows):
    rows = []
    for global_row, joint_row in zip(global_rows, joint_rows):
        if not global_row['valid']:
            continue
        a, b = global_row['top1_correct'], joint_row['top1_correct']
        status = 'both_correct' if a and b else 'both_wrong' if not a and not b else 'repair' if b else 'harm'
        rows.append({'query_index': global_row['query_index'], 'status': status,
                     'global_AP': global_row['AP'], 'joint_AP': joint_row['AP'],
                     'AP_delta': joint_row['AP']-global_row['AP'],
                     'global_first_positive_rank': global_row['first_positive_rank'],
                     'joint_first_positive_rank': joint_row['first_positive_rank']})
    summary = {status: sum(row['status'] == status for row in rows)
               for status in ('repair', 'harm', 'both_correct', 'both_wrong')}
    summary.update(valid_queries=len(rows),
                   mean_AP_delta=float(np.mean([r['AP_delta'] for r in rows])),
                   net_top1_change=summary['repair']-summary['harm'])
    return summary, rows


@torch.no_grad()
def evaluate(cfg, model, loader, num_query, device, include_diagnostics=True):
    if cfg.TEST.RE_RANKING or cfg.TEST.FEAT_NORM != 'yes':
        raise ValueError('This diagnostic protocol uses L2 normalization without reranking')
    model.eval()
    names = ['retrieval', 'mldc_after_bn', 'lpam_after_bn']
    if include_diagnostics:
        names += ['raw_readout', 'raw_last_cls', 'raw_uniform_cls', 'raw_lpam']
    batches = {name: [] for name in names}
    pid_list, camera_list = [], []
    depth_sum = torch.zeros(model.lpam.queries, len(model.patch_layers), dtype=torch.float64)
    layer_norms = torch.zeros(12, dtype=torch.float64)
    cosine, overlap, count = 0., 0., 0
    for images, pids, cams, camera, view, _ in loader:
        parts = model(images.to(device), cam_label=camera.to(device), view_label=view.to(device), return_components=True)
        for name in names:
            batches[name].append(parts[name].float().cpu())
        pid_list.extend(int(p) for p in pids)
        camera_list.extend(int(c) for c in cams)
        depth_sum += parts['depth_weights'].double().sum(0).cpu()
        layer_norms += parts['layers'].float().norm(dim=-1).double().sum(1).cpu()
        cosine += F.cosine_similarity(parts['raw_readout'].float(), parts['raw_last_cls'].float()).sum().item()
        overlap += parts['overlap_loss'].item()*len(images)
        count += len(images)
    features = {name: torch.cat(values) for name, values in batches.items()}
    pids, cameras = np.asarray(pid_list), np.asarray(camera_list)
    metrics, rankings = {}, {}
    for name, vectors in features.items():
        metrics[name], rankings[name] = rank_features(vectors, pids, cameras, num_query)
    local_parts = features['lpam_after_bn'].reshape(count, model.lpam.queries, model.lpam.dim)
    if include_diagnostics:
        for i in range(model.lpam.queries):
            name = 'local_part_{}'.format(i+1)
            metrics[name], rankings[name] = rank_features(local_parts[:, i], pids, cameras, num_query)
    summary, query_changes = complementarity(rankings['mldc_after_bn'], rankings['retrieval'])
    # Same checkpoint, removing one part and renormalizing the remaining score.
    removal = {}
    if include_diagnostics:
        for i in range(model.lpam.queries):
            reduced = features['retrieval'].clone()
            offset = model.base.in_planes+i*model.lpam.dim
            reduced[:, offset:offset+model.lpam.dim] = 0
            removal[str(i+1)], _ = rank_features(reduced, pids, cameras, num_query)
    diagnostics = {'readouts': {k: v for k, v in metrics.items() if k != 'retrieval'},
                   'complementarity': summary, 'part_removal': removal,
                   'mean_depth_weights': (depth_sum/count).tolist(),
                   'mean_attention_overlap_squared_cosine': overlap/count,
                   'mean_raw_layer_norms': (layer_norms/count).tolist(),
                   'layer_weights': model.readout_weights().detach().cpu().tolist(),
                   'raw_readout_last_cls_cosine_mean': cosine/count,
                   'local_similarity_weight': model.local_weight,
                   'descriptor_dim': model.descriptor_dim,
                   'note': 'All readouts and part removals are same-checkpoint diagnostics, not independent ablations.'}
    cache = {'features': {k: features[k] for k in names[:3]}, 'pids': pids, 'cameras': cameras,
             'num_query': num_query, 'rankings': rankings, 'query_changes': query_changes}
    return metrics['retrieval'], diagnostics, cache


def append_csv(path, rows):
    if not rows:
        return
    with Path(path).open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)


def record_evaluation(output, epoch, metrics, diagnostics, cache):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    append_csv(output/'evaluation.csv', [{'epoch': epoch, **metrics}])
    append_csv(output/'diagnostics.csv', [{'epoch': epoch, 'feature': name, **row}
                                         for name, row in diagnostics['readouts'].items()])
    with (output/'evaluation_diagnostics.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({'epoch': epoch, 'retrieval': metrics, **diagnostics})+'\n')
    destination = output/'query_diagnostics'/'epoch_{:03d}'.format(epoch)
    destination.mkdir(parents=True, exist_ok=True)
    for name, rows in cache['rankings'].items():
        with (destination/(name+'.csv')).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows([{**row, 'top10': json.dumps(row['top10'])} for row in rows])
    with (destination/'complementarity.csv').open('w', newline='', encoding='utf-8') as handle:
        rows = cache['query_changes']
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (destination/'summary.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
