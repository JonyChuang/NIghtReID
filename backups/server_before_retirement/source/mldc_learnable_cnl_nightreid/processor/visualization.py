"""Fixed-pair retrieval attribution and learned layer-weight plots."""
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def fixed_pairs(records, num_query, count, seed):
    gallery = {}
    for index in range(num_query, len(records)):
        gallery.setdefault(records[index][1], []).append(index)
    candidates = list(range(num_query))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    pairs, identities = [], set()
    for query in candidates:
        pid, cam = records[query][1:3]
        positives = [i for i in gallery.get(pid, []) if records[i][2] != cam]
        if not positives or pid in identities:
            continue
        reference = rng.choice(positives)
        pairs.append({'query_index': query, 'reference_index': reference,
                      'pid': int(pid), 'query_camera': int(cam),
                      'reference_camera': int(records[reference][2]),
                      'query_path': str(records[query][0]),
                      'reference_path': str(records[reference][0])})
        identities.add(pid)
        if len(pairs) >= count:
            break
    return pairs


def attribution(feature, reference, patch_tokens, grid):
    score = F.cosine_similarity(feature.float(), reference.detach().float(), dim=1).sum()
    gradient = torch.autograd.grad(score, patch_tokens, retain_graph=True)[0]
    values = (patch_tokens.float() * gradient.float()).abs().mean(-1)
    return values.reshape(*grid).detach().cpu().numpy(), float(score.detach())


def spatial_share(values):
    return values / max(float(values.sum()), 1e-12)


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_weights(model, output, epoch):
    plt = _plt()
    path = Path(output)
    with (path / 'layer_weights.csv').open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    history = np.asarray([[float(row[f'layer_{i}']) for i in range(1, 13)] for row in rows])
    epochs = [int(row['epoch']) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].bar(np.arange(1, 13), model.readout_weights().detach().cpu().numpy(), color='#397699')
    axes[0].set(xlabel='Transformer layer', ylabel='Readout weight',
                title=f'Epoch {epoch} | {model.readout}', xticks=np.arange(1, 13), ylim=(0, 1))
    display = axes[1].imshow(history.T, origin='lower', aspect='auto', vmin=0, vmax=1, cmap='viridis')
    ticks = np.unique(np.linspace(0, len(epochs)-1, min(6, len(epochs))).astype(int))
    axes[1].set(xticks=ticks, xticklabels=[epochs[i] for i in ticks],
                yticks=np.arange(12), yticklabels=np.arange(1, 13), xlabel='Epoch',
                ylabel='Transformer layer', title='Readout weights over training')
    fig.colorbar(display, ax=axes[1], label='Readout weight')
    destination = path / 'visualizations' / f'epoch_{epoch:03d}'
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / 'layer_weights.png', dpi=150)
    fig.savefig(path / 'layer_weights.png', dpi=150)
    plt.close(fig)


def plot_metrics(output):
    path = Path(output)
    if not (path / 'evaluation.csv').exists():
        return
    plt = _plt()
    with (path / 'evaluation.csv').open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for metric in ('mAP', 'rank1'):
        axes[0].plot([int(r['epoch']) for r in rows], [100*float(r[metric]) for r in rows], 'o-', label=metric)
    axes[0].set(xlabel='Epoch', ylabel='Retrieval (%)', title='Trained readout: configured retrieval space')
    if (path / 'diagnostics.csv').exists():
        with (path / 'diagnostics.csv').open(encoding='utf-8') as handle:
            diagnostics = list(csv.DictReader(handle))
        for readout, label in (('raw_readout', 'MLDC/readout'), ('raw_last_cls', 'Last CLS'),
                               ('raw_uniform_cls', 'Uniform CLS average')):
            selected = [r for r in diagnostics if r['feature'] == readout]
            axes[1].plot([int(r['epoch']) for r in selected], [100*float(r['mAP']) for r in selected], 'o-', label=label)
    axes[1].set(xlabel='Epoch', ylabel='mAP (%)', title='Same checkpoint diagnostics: all before BN')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(path / 'retrieval_curves.png', dpi=150)
    plt.close(fig)


def _render_case(destination, query_path, reference_path, maps, scores, epoch, readout, neck, correlation, retrieval_mode='mldc'):
    from PIL import Image
    plt = _plt()
    with Image.open(query_path) as source:
        query = source.convert('RGB')
    with Image.open(reference_path) as source:
        reference = source.convert('RGB')
    width, height = query.size
    def resize(values):
        return F.interpolate(torch.as_tensor(values).float()[None, None], size=(height, width),
                             mode='bicubic', align_corners=False)[0, 0].clamp_min(0).numpy()
    last, current, retrieval = [resize(values) for values in maps]
    shared_upper = max(float(np.percentile(np.concatenate((last.ravel(), current.ravel())), 99)), 1e-12)
    retrieval_upper = max(float(np.percentile(retrieval, 99)), 1e-12)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 8), constrained_layout=True)
    axes[0, 0].imshow(query)
    axes[0, 0].set_title('Fixed night query')
    axes[1, 0].imshow(reference)
    axes[1, 0].set_title('Same ID | different camera')
    retrieval_title = ('Retrieval | MLDC + last CLS\nBefore BN; own color scale'
                       if retrieval_mode == 'raw_residual' else f'Retrieval | {neck} BN\nOwn color scale')
    if retrieval_mode == 'supervised_residual':
        retrieval_title = 'Retrieval | (MLDC + last CLS) / 2\nBefore BN; own color scale'
    panels = [(axes[0, 1], last, shared_upper, 'Last CLS | before BN', scores[0]),
              (axes[0, 2], current, shared_upper, f'{readout} | before BN', scores[1]),
              (axes[1, 1], retrieval, retrieval_upper, retrieval_title, scores[2])]
    for ax, values, upper, title, score in panels:
        intensity = np.clip(values / upper, 0, 1)
        colors = plt.get_cmap('turbo')(intensity)
        colors[..., 3] = .65 * intensity**.7
        ax.imshow(query)
        ax.imshow(colors, extent=(0, width, height, 0))
        ax.set_title(f'{title}\nPositive cosine {score:.3f}', fontsize=10)
    delta = spatial_share(maps[1]) - spatial_share(maps[0])
    diff = axes[1, 2].imshow(delta, cmap='RdBu_r', vmin=-.03, vmax=.03, interpolation='nearest')
    axes[1, 2].set_title('Change in spatial attribution share\nReadout minus last CLS; fixed scale', fontsize=10)
    fig.colorbar(diff, ax=axes[1, 2], shrink=.7, label='Share per patch')
    for ax in axes.ravel():
        ax.axis('off')
    fig.suptitle(f'Epoch {epoch} | raw-map Pearson {correlation:.4f}\n'
                 'Same checkpoint; |gradient x activation| at patch embeddings\n'
                 'Top heatmaps share their color scale; differences do not measure retrieval gain', fontsize=10)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def save_visualizations(cfg, model, loader, num_query, epoch, logger):
    output = Path(cfg.OUTPUT_DIR)
    plot_weights(model, output, epoch)
    plot_metrics(output)
    records = loader.dataset.dataset
    pairs = fixed_pairs(records, num_query, cfg.TEST.MLDC_VIS_SAMPLES, int(cfg.SOLVER.SEED))
    if not pairs:
        raise ValueError('No cross-camera positive pairs available for MLDC visualization')
    pair_path = output / 'visualization_pairs.json'
    if pair_path.exists() and json.loads(pair_path.read_text(encoding='utf-8')) != pairs:
        raise ValueError('Fixed visualization pairs changed; use a separate output directory')
    pair_path.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding='utf-8')
    destination = output / 'visualizations' / f'epoch_{epoch:03d}'
    destination.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    grid = (model.base.patch_embed.num_y, model.base.patch_embed.num_x)
    was_training = model.training
    rng_states = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                  torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
    model.eval()
    try:
        for pair in pairs:
            qi, ri = pair['query_index'], pair['reference_index']
            def inputs(index):
                image, _, camera, view, _ = loader.dataset[index]
                return (image.unsqueeze(0).to(device), torch.tensor([camera], device=device),
                        torch.tensor([view], device=device))
            reference_image, reference_camera, reference_view = inputs(ri)
            with torch.no_grad():
                ref_output, ref_raw, ref_layers = model(reference_image, cam_label=reference_camera,
                                                       view_label=reference_view, return_components=True)
            captured = []
            def capture(module, args, result):
                if not result.requires_grad:
                    result = result.detach().requires_grad_(True)
                captured.append(result)
                return result
            handle = model.base.patch_embed.register_forward_hook(capture)
            try:
                query_image, query_camera, query_view = inputs(qi)
                with torch.enable_grad():
                    query_output, query_raw, query_layers = model(query_image, cam_label=query_camera,
                                                                 view_label=query_view, return_components=True)
                    maps_and_scores = [attribution(a, b, captured[0], grid) for a, b in
                                       ((query_layers[-1], ref_layers[-1]), (query_raw, ref_raw),
                                        (query_output, ref_output))]
                maps, scores = zip(*maps_and_scores)
                correlation = (float(np.corrcoef(maps[0].ravel(), maps[1].ravel())[0, 1])
                               if np.std(maps[0]) > 0 and np.std(maps[1]) > 0 else float('nan'))
                descriptor_cosine = float(F.cosine_similarity(query_layers[-1].detach(), query_raw.detach()).item())
                topk = min(16, maps[0].size)
                overlap = len(set(maps[0].ravel().argsort()[-topk:]) & set(maps[1].ravel().argsort()[-topk:])) / topk
                stem = f'query_{qi:05d}_pid_{pair["pid"]}'
                np.savez_compressed(destination / (stem + '.npz'),
                                    last_cls_attribution=maps[0], readout_attribution=maps[1],
                                    retrieval_attribution=maps[2],
                                    normalized_attribution_delta=spatial_share(maps[1])-spatial_share(maps[0]),
                                    layer_weights=model.readout_weights().detach().cpu().numpy(),
                                    layer_cls_norms=query_layers.detach().float().norm(dim=-1)[:, 0].cpu().numpy(),
                                    patch_grid=np.asarray(grid), epoch=epoch, **pair,
                                    readout=model.readout, neck_feat=model.neck_feat,
                                    retrieval_mode=getattr(model, 'retrieval', 'mldc'),
                                    training_feature=getattr(model, 'train_feature', 'mldc'),
                                    last_positive_cosine=scores[0], readout_positive_cosine=scores[1],
                                    retrieval_positive_cosine=scores[2], raw_descriptor_cosine=descriptor_cosine,
                                    attribution_pearson=correlation, top16_overlap=overlap)
                _render_case(destination / (stem + '.png'), pair['query_path'], pair['reference_path'],
                             maps, scores, epoch, model.readout, model.neck_feat, correlation,
                             getattr(model, 'retrieval', 'mldc'))
                stats_path = output / 'heatmap_statistics.csv'
                with stats_path.open('a', newline='', encoding='utf-8') as stream:
                    writer = csv.writer(stream)
                    if stream.tell() == 0:
                        writer.writerow(['epoch', 'query_index', 'reference_index', 'pid', 'attribution_pearson',
                                         'top16_overlap', 'raw_descriptor_cosine', 'last_positive_cosine',
                                         'readout_positive_cosine', 'retrieval_positive_cosine'])
                    writer.writerow([epoch, qi, ri, pair['pid'], correlation, overlap, descriptor_cosine, *scores])
            finally:
                handle.remove()
        logger.info('Saved %d fixed-pair heatmaps, raw NPZ and layer-weight plots to %s', len(pairs), destination)
    finally:
        model.train(was_training)
        random.setstate(rng_states[0])
        np.random.set_state(rng_states[1])
        torch.set_rng_state(rng_states[2])
        if rng_states[3] is not None:
            torch.cuda.set_rng_state_all(rng_states[3])
