"""Fixed-query locality diagnostics, Gaussian maps and retrieval attribution."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .visualization import save_visualizations as save_baseline_visualizations
from .visualization import fixed_pairs, _plt, attribution, spatial_share


def locality_statistics(tokens, grid):
    """Eight-neighbor cos, all-patch cos, and patch/CLS cos (exclude self)."""
    unit = F.normalize(tokens.detach().float(), dim=-1)
    patches = unit[:, 1:]
    count = patches.shape[1]
    rows, cols = torch.meshgrid(torch.arange(grid[0], device=tokens.device),
                               torch.arange(grid[1], device=tokens.device), indexing='ij')
    coordinates = torch.stack((rows.flatten(), cols.flatten()), -1)
    distance = (coordinates[:, None] - coordinates[None, :]).abs().amax(-1)
    neighbors = (distance == 1)
    similarities = patches @ patches.transpose(-2, -1)
    # Average each patch's valid neighborhood first; corners are not downweighted.
    local = ((similarities * neighbors).sum(-1) / neighbors.sum(-1)).mean()
    global_cosine = ((similarities.sum((-2, -1)) - count) / (count * (count - 1))).mean()
    cls_cosine = (patches * unit[:, :1]).sum(-1).mean()
    return {'neighbor_cosine': float(local), 'global_patch_cosine': float(global_cosine),
            'patch_cls_cosine': float(cls_cosine), 'local_global_gap': float(local - global_cosine)}


def capture_locality(model, image, camera, view):
    rows, handles = [], []
    grid = (model.base.patch_embed.num_y, model.base.patch_embed.num_x)
    def capture(index):
        def hook(module, inputs, outputs):
            rows.append({'layer': index + 1, **locality_statistics(outputs, grid)})
        return hook
    try:
        for index, block in enumerate(model.base.blocks):
            handles.append(block.register_forward_hook(capture(index)))
            if hasattr(block.attn, 'gaug'):
                block.attn.capture = block.attn.gaug.capture = True
        model.prr.capture = True
        with torch.no_grad():
            detail = model(image, cam_label=camera, view_label=view, return_details=True)
        arrays = {}
        for index, block in enumerate(model.base.blocks):
            attention = block.attn
            if hasattr(attention, 'gaug'):
                stats = attention.gaug.diagnostics
                var = stats['variance'].cpu().numpy()
                alpha = stats['alpha'].cpu().numpy()
                rows[index].update(variance_y=float(var[..., 0].mean()),
                                   variance_x=float(var[..., 1].mean()),
                                   alpha_mean=float(alpha.mean()), alpha_std=float(alpha.std()))
                for name, value in stats.items():
                    arrays[f'layer_{index+1}_{name}'] = value.cpu().numpy()
                if index + 1 in [3, 6, 10, 12]:
                    for name, value in attention.diagnostics.items():
                        arrays[f'layer_{index+1}_{name}'] = value.cpu().numpy()
        prr_stats = {}
        if model.locat_spec['prr']:
            weights = model.prr.attention.float().cpu()
            arrays['prr_cls_attention'] = weights.numpy()
            prr_stats = {'cls_self_mass': float(weights[..., 0].mean()),
                         'patch_mass': float(weights[..., 1:].sum(-1).mean()),
                         'normalized_entropy': float((-(weights * weights.clamp_min(1e-12).log()).sum(-1) /
                                                       np.log(weights.shape[-1])).mean()),
                         'readout_mldc_cosine': float(F.cosine_similarity(
                             detail['raw_readout'].float(), detail['raw_mldc'].float()).mean())}
        return rows, arrays, prr_stats
    finally:
        for handle in handles:
            handle.remove()
        for block in model.base.blocks:
            if hasattr(block.attn, 'gaug'):
                block.attn.capture = block.attn.gaug.capture = False
                block.attn.diagnostics.clear()
                block.attn.gaug.diagnostics.clear()
        model.prr.capture = False
        model.prr.attention = None


def render_locality(destination, image, rows, arrays, grid, label):
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Fixed night query')
    axes[0, 0].axis('off')
    layers = [row['layer'] for row in rows]
    for key, name in [('neighbor_cosine', '8-neighbor'), ('global_patch_cosine', 'All other patches'),
                      ('patch_cls_cosine', 'Patch / CLS')]:
        axes[0, 1].plot(layers, [row[key] for row in rows], 'o-', label=name)
    axes[0, 1].set(xlabel='Layer', ylabel='Cosine', title='Locality and homogenization')
    axes[0, 1].legend(fontsize=8)
    axes[0, 2].plot(layers, [row['local_global_gap'] for row in rows], 'o-')
    axes[0, 2].set(xlabel='Layer', ylabel='Neighbor minus global cosine', title='Local / global separation')
    if 'variance_y' in rows[0]:
        for key, name in [('variance_y', 'Vertical variance'), ('variance_x', 'Horizontal variance'),
                          ('alpha_mean', 'Alpha')]:
            axes[1, 0].plot(layers, [row[key] for row in rows], 'o-', label=name)
        axes[1, 0].set(xlabel='Layer', title='Learned Gaussian parameters')
        axes[1, 0].legend(fontsize=8)
        patch_index = (grid[0] // 2) * grid[1] + grid[1] // 2
        content = arrays['layer_10_content_attention'][0, :, patch_index+1, 1:].mean(0).reshape(grid)
        augmented = arrays['layer_10_augmented_attention'][0, :, patch_index+1, 1:].mean(0).reshape(grid)
        maximum = max(float(content.max()), float(augmented.max()), 1e-12)
        for ax, values, title in [(axes[1, 1], content, 'L10 content attention'),
                                  (axes[1, 2], augmented, 'L10 attention + Gaussian')]:
            plot = ax.imshow(values, cmap='viridis', vmin=0, vmax=maximum)
            ax.scatter([grid[1] // 2], [grid[0] // 2], marker='x', color='red')
            ax.set_title(title + '\nSame Q/K, shared scale', fontsize=10)
        fig.colorbar(plot, ax=[axes[1, 1], axes[1, 2]], shrink=.7, label='Attention probability')
    else:
        for ax in axes[1]:
            ax.axis('off')
    fig.suptitle(label + '\nFixed query diagnostics; map changes alone do not establish retrieval improvement', fontsize=11)
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def render_attribution(destination, image, maps, scores, prr_attention, grid):
    plt = _plt()
    fig, axes = plt.subplots(1, 4, figsize=(12, 5), constrained_layout=True)
    height, width = image.shape[:2]
    maximum = max(float(np.percentile(np.concatenate([m.ravel() for m in maps]), 99)), 1e-12)
    for ax, values, name, score in zip(axes[:3], maps, ['Raw MLDC', 'LocAt before BN', 'Retrieval after BN'], scores):
        resized = F.interpolate(torch.from_numpy(values)[None, None].float(), size=(height, width),
                                mode='bilinear', align_corners=False)[0, 0].numpy()
        # All three heatmaps share the same scale.
        intensity = np.clip(resized / maximum, 0, 1)
        colors = plt.get_cmap('turbo')(intensity)
        colors[..., 3] = .65 * intensity ** .7
        ax.imshow(image)
        ax.imshow(colors)
        ax.set_title('{}\nPositive cosine {:.3f}'.format(name, score), fontsize=10)
        ax.axis('off')
    if prr_attention is not None:
        weights = prr_attention[0, :, 0, 1:].mean(0).reshape(grid)
        plot = axes[3].imshow(weights, cmap='viridis', vmin=0)
        axes[3].set_title('PRR CLS -> patch attention\nHead mean; separate scale', fontsize=10)
        fig.colorbar(plot, ax=axes[3], shrink=.7)
    else:
        axes[3].axis('off')
    fig.suptitle('Same query/reference pair | |gradient x activation| at patch embeddings\n'
                 'Before/after diagnostics within this checkpoint; not an independent baseline comparison', fontsize=10)
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def save_visualizations(cfg, model, loader, num_query, epoch, logger):
    save_baseline_visualizations(cfg, model, loader, num_query, epoch, logger)
    output = Path(cfg.OUTPUT_DIR)
    destination = output / 'visualizations' / f'epoch_{epoch:03d}'
    pairs = fixed_pairs(loader.dataset.dataset, num_query, cfg.TEST.MLDC_VIS_SAMPLES, int(cfg.SOLVER.SEED))
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        for pair in pairs:
            qi, ri = pair['query_index'], pair['reference_index']
            def inputs(index):
                image, _, camera, view, _ = loader.dataset[index]
                return image.unsqueeze(0).to(device), torch.tensor([camera], device=device), torch.tensor([view], device=device)
            image, camera, view = inputs(qi)
            grid = (model.base.patch_embed.num_y, model.base.patch_embed.num_x)
            rows, arrays, prr_stats = capture_locality(model, image, camera, view)
            stem = f'query_{qi:05d}_pid_{pair["pid"]}'
            with torch.no_grad():
                ref_image, ref_cam, ref_view = inputs(ri)
                reference = model(ref_image, cam_label=ref_cam, view_label=ref_view, return_details=True)
            captured = []
            def hook(module, args, result):
                captured.append(result)
            handle = model.base.patch_embed.register_forward_hook(hook)
            try:
                with torch.enable_grad():
                    detail = model(image, cam_label=camera, view_label=view, return_details=True)
                    map_scores = [attribution(detail[name], reference[name], captured[0], grid)
                                  for name in ['raw_mldc', 'raw_readout', 'retrieval']]
            finally:
                handle.remove()
            maps, scores = zip(*map_scores)
            for name, values in zip(['mldc_attribution', 'locat_attribution', 'retrieval_attribution'], maps):
                arrays[name] = values
            arrays['attribution_share_delta'] = spatial_share(maps[1]) - spatial_share(maps[0])
            np.savez_compressed(destination / (stem + '_locat.npz'), **arrays)
            (destination / (stem + '_locality.json')).write_text(json.dumps({
                'pair': pair, 'epoch': epoch, 'locat_spec': model.locat_spec,
                'layers': rows, 'prr': prr_stats, 'positive_cosines': list(scores),
                'scope': 'Fixed-query diagnostics; no foreground annotation or external teacher used.'
            }, indent=2), encoding='utf-8')
            image_rgb = ((image[0].detach().cpu().permute(1, 2, 0).numpy() *
                          np.asarray(cfg.INPUT.PIXEL_STD)) + np.asarray(cfg.INPUT.PIXEL_MEAN)).clip(0, 1)
            render_locality(destination / (stem + '_locality.png'), image_rgb, rows, arrays, grid,
                            f'Epoch {epoch} | {model.locat_spec}')
            render_attribution(destination / (stem + '_locat_attribution.png'), image_rgb, maps, scores,
                               arrays.get('prr_cls_attention'), grid)
            path = output / 'locality_statistics.csv'
            with path.open('a', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=['epoch', 'query_index', *rows[0].keys()])
                if handle.tell() == 0:
                    writer.writeheader()
                for row in rows:
                    writer.writerow({'epoch': epoch, 'query_index': qi, **row})
        logger.info('Saved LocAt locality, Gaussian and MLDC/LocAt attribution diagnostics for %d fixed queries', len(pairs))
    finally:
        model.train(was_training)
