"""Patch attention allocation; these maps are not causal identity attribution."""
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def save_visualizations(cfg, model, loader, num_query, epoch, logger):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    output = Path(cfg.OUTPUT_DIR)
    destination = output / 'visualizations' / 'epoch_{:03d}'.format(epoch)
    destination.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(1, 13), model.readout_weights().detach().cpu().numpy())
    ax.set(xticks=range(1, 13), xlabel='Layer', ylabel='Frozen CLS weight',
           title='Deep-four CLS anchor; weights are frozen')
    fig.tight_layout()
    fig.savefig(destination / 'cls_anchor_weights.png', dpi=130)
    plt.close(fig)
    if (output / 'evaluation.csv').is_file():
        with (output / 'evaluation.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        fig, ax = plt.subplots(figsize=(7, 3))
        for key in ('mAP', 'rank1'):
            ax.plot([int(r['epoch']) for r in rows], [100 * float(r[key]) for r in rows], label=key)
        ax.set(xlabel='Epoch', ylabel='Retrieval (%)', title='Compensated BN descriptor')
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / 'retrieval_curves.png', dpi=130)
        plt.close(fig)
    records = loader.dataset.dataset
    indices = list(range(min(num_query, len(records))))
    random.Random(int(cfg.SOLVER.SEED)).shuffle(indices)
    indices = indices[:int(cfg.TEST.MLDC_VIS_SAMPLES)]
    device = next(model.parameters()).device
    grid = (model.base.patch_embed.num_y, model.base.patch_embed.num_x)
    was_training = model.training
    model.eval()
    try:
        for index in indices:
            image, _, camera, view, _ = loader.dataset[index]
            with torch.no_grad():
                data = model(image.unsqueeze(0).to(device),
                             cam_label=torch.tensor([camera], device=device),
                             view_label=torch.tensor([view], device=device),
                             return_patch_diagnostics=True)
            maps = data['patch_attention'][0].mean(0).reshape(-1, *grid).cpu().numpy()
            masses = data['layer_mass'][0].mean(0).cpu().numpy()
            nulls = data['null_mass'][0].cpu().tolist()
            with Image.open(records[index][0]) as source:
                original = source.convert('RGB')
            width, height = original.size
            layers = model.patch_compensation.layers
            columns = 5
            rows = (len(layers) + 2 + columns - 1) // columns
            fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.4, rows * 3.5))
            axes = np.asarray(axes).reshape(-1)
            axes[0].imshow(original)
            axes[0].set_title('Query {}'.format(index))
            upper = max(float(maps.max()), 1e-12)
            for i, layer in enumerate(layers):
                resized = F.interpolate(torch.from_numpy(maps[i])[None, None],
                                        size=(height, width), mode='bilinear', align_corners=False)[0, 0].numpy()
                colors = plt.get_cmap('turbo')(np.clip(resized / upper, 0, 1))
                colors[..., 3] = 0.65 * np.clip(resized / upper, 0, 1)
                axes[i + 1].imshow(original)
                axes[i + 1].imshow(colors)
                axes[i + 1].set_title('L{} | mass {:.3f}'.format(layer, masses[i]))
            bar = axes[len(layers) + 1]
            bar.bar(list(map(str, layers)) + ['null'], list(masses) + [float(np.mean(nulls))])
            bar.set_title('Attention mass')
            for ax in axes:
                if ax is not bar:
                    ax.axis('off')
            fig.suptitle('Epoch {} | head-mean attention; shared scale\n'
                         'Allocation does not establish identity evidence'.format(epoch))
            fig.tight_layout(rect=(0, 0, 1, .94))
            stem = 'query_{:05d}'.format(index)
            fig.savefig(destination / (stem + '.png'), dpi=130)
            plt.close(fig)
            np.savez_compressed(destination / (stem + '.npz'), attention=maps,
                                layer_mass=masses, null_mass=np.asarray(nulls),
                                patch_layers=np.asarray(layers), patch_grid=np.asarray(grid))
            ratio = (data['delta'].float().norm() / data['anchor'].float().norm().clamp_min(1e-12)).item()
            (destination / (stem + '.json')).write_text(json.dumps({
                'query_index': index, 'epoch': epoch, 'null_mass_per_head': nulls,
                'patch_layers': list(layers), 'layer_mass': masses.tolist(),
                'delta_anchor_norm_ratio': ratio,
                'note': 'Attention allocation, not a measure of identity contribution.'}, indent=2))
    finally:
        model.train(was_training)
    logger.info('Saved %d patch-allocation visualizations in %s', len(indices), destination)
