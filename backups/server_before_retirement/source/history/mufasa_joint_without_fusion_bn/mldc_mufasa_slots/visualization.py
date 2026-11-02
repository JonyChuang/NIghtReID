"""Same fixed cross-camera image pairs at every recorded epoch."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from mldc_cls_readout.processor.visualization import fixed_pairs


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_curves(output):
    output = Path(output)
    path = output / 'evaluation.csv'
    if not path.exists():
        return
    with path.open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, metric in zip(axes, ['mAP', 'rank1']):
        for name in ['mldc', 'final', 'residual_only']:
            chosen = [r for r in rows if r['feature'] == name]
            ax.plot([int(r['epoch']) for r in chosen], [100*float(r[metric]) for r in chosen],
                    'o-', label=name)
        ax.set(xlabel='Epoch', ylabel=metric+' (%)')
        ax.legend()
        ax.grid(alpha=.2)
    fig.savefig(output / 'retrieval_curves.png', dpi=150)
    plt.close(fig)


@torch.no_grad()
def save_visualizations(cfg, model, loader, num_query, epoch):
    from PIL import Image
    output = Path(cfg.OUTPUT_DIR)
    records = loader.dataset.dataset
    # Seed independent of training seed, retaining the six prior MLDC pairs.
    pairs = fixed_pairs(records, num_query, cfg.MUFASA.VIS_SAMPLES, 1234)
    path = output / 'visualization_pairs.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != pairs:
        raise ValueError('Visualization records changed')
    path.write_text(json.dumps(pairs, indent=2), encoding='utf-8')
    destination = output / 'visualizations' / f'epoch_{epoch:03d}'
    destination.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    plt = pyplot()
    try:
        for pair in pairs:
            indices = [pair['query_index'], pair['reference_index']]
            items = [loader.dataset[i] for i in indices]
            images = torch.stack([item[0] for item in items]).to(device)
            cams = torch.tensor([item[2] for item in items], device=device)
            views = torch.tensor([item[3] for item in items], device=device)
            parts = model(images, cam_label=cams, view_label=views, return_parts=True, reconstruct=False)
            maps = parts['maps'].cpu().numpy()
            raw = parts['raw_maps'].cpu().numpy()
            fused = parts['fused_map'].cpu().numpy()
            attention = parts['readout_attention'].cpu().numpy()
            scales = parts['fusion_scale'].cpu().numpy()
            stem = f"q{pair['query_index']:05d}"
            np.savez_compressed(destination / (stem+'.npz'), aligned_maps=maps, raw_maps=raw,
                                fused_map=fused, readout_attention=attention,
                                orders=parts['orders'].cpu().numpy(), fusion_scale=scales,
                                grid=np.asarray(model.grid))
            fig, axes = plt.subplots(2, 6, figsize=(14, 7), constrained_layout=True)
            colors = plt.get_cmap('tab10', cfg.MUFASA.NUM_SLOTS)
            for row, item in enumerate(items):
                with Image.open(item[4]) as source:
                    image = np.asarray(source.convert('RGB').resize((128, 256)))
                axes[row, 0].imshow(image)
                axes[row, 0].set_title('Query' if row == 0 else 'Same ID, other camera')
                for branch, layer in enumerate(model.layers):
                    labels = maps[row, branch].argmax(-1).reshape(model.grid)
                    axes[row, branch+1].imshow(image)
                    axes[row, branch+1].imshow(labels, cmap=colors, vmin=0, vmax=cfg.MUFASA.NUM_SLOTS-1,
                                               extent=(0, 128, 256, 0), alpha=.6, interpolation='nearest')
                    axes[row, branch+1].set_title(f'Block {layer}: aligned SA')
                weighted = (fused[row]*attention[row][None, :]).sum(-1).reshape(model.grid)
                axes[row, 5].imshow(image)
                axes[row, 5].imshow(weighted, cmap='magma', extent=(0, 128, 256, 0),
                                   alpha=.6, interpolation='nearest', vmin=0, vmax=1)
                weight_lines = [' '.join(f'{w:.3f}' for w in attention[row, i:i+2])
                                for i in range(0, len(attention[row]), 2)]
                axes[row, 5].set_title(f'Weighted SA | gate {scales[row]:.3f}\n'+'\n'.join(weight_lines), fontsize=9)
                for ax in axes[row]:
                    ax.axis('off')
            fig.suptitle(f'Epoch {epoch} | pid {pair["pid"]} | SA maps are not attribution or part labels\n'
                         'Slot colors are aligned within each image; they need not correspond across images.')
            fig.savefig(destination / (stem+'.png'), dpi=140)
            plt.close(fig)
    finally:
        model.train(was_training)
