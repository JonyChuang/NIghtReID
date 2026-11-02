"""Fixed-pair attention pictures without extra evaluation or backward passes."""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def fixed_pairs(records, num_query, count, seed):
    gallery = {}
    for i in range(num_query, len(records)):
        gallery.setdefault(records[i][1], []).append(i)
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
                      'query_path': str(records[query][0]), 'reference_path': str(records[reference][0])})
        identities.add(pid)
        if len(pairs) >= count:
            break
    return pairs


def plt_module():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def image_array(dataset, index, cfg):
    image = dataset[index][0].float().permute(1, 2, 0).numpy()
    return np.clip(image*np.asarray(cfg.INPUT.PIXEL_STD)+np.asarray(cfg.INPUT.PIXEL_MEAN), 0, 1)


def image_batch(dataset, indices, device):
    items = [dataset[i] for i in indices]
    return (torch.stack([item[0] for item in items]).to(device),
            torch.tensor([item[2] for item in items], device=device),
            torch.tensor([item[3] for item in items], device=device))


def overlay(axis, picture, values, title, signed=False, bound=None):
    axis.imshow(picture)
    resized = F.interpolate(torch.as_tensor(values.copy())[None, None].float(), size=picture.shape[:2],
                            mode='bilinear', align_corners=False)[0, 0].numpy()
    if signed:
        bound = max(float(bound or np.abs(values).max()), 1e-10)
        display = axis.imshow(resized, alpha=.65, cmap='RdBu_r', vmin=-bound, vmax=bound)
    else:
        display = axis.imshow(resized, alpha=.6, cmap='magma', vmin=0,
                              vmax=max(float(bound or values.max()), 1e-10))
    axis.set_title(title, fontsize=9)
    axis.axis('off')
    return display


@torch.no_grad()
def save_visualizations(cfg, model, loader, num_query, epoch, logger):
    """Only fixed-pair attention pictures; no ranking, ablation or gradients."""
    model.eval()
    output = Path(cfg.OUTPUT_DIR)
    destination = output/'visualizations'/'epoch_{:03d}'.format(epoch)
    destination.mkdir(parents=True, exist_ok=True)
    pairs = fixed_pairs(loader.dataset.dataset, num_query, cfg.TEST.MLDC_VIS_SAMPLES, cfg.SOLVER.SEED)
    (output/'fixed_visualization_pairs.json').write_text(json.dumps(pairs, indent=2), encoding='utf-8')
    device = next(model.parameters()).device
    plt = plt_module()
    for pair in pairs:
        indices = [pair['query_index'], pair['reference_index']]
        images, camera, view = image_batch(loader.dataset, indices, device)
        parts = model(images, cam_label=camera, view_label=view, return_components=True)
        attention = parts['attention'].float().cpu().numpy().reshape(2, model.lpam.queries, *parts['grid_hw'])
        depths = parts['depth_weights'].float().cpu().numpy()
        weights = parts['fusion_weights'].float().cpu().numpy()
        fig, axes = plt.subplots(2, model.lpam.queries+1, figsize=(14, 6), constrained_layout=True)
        bound = float(attention.max())
        for row, index in enumerate(indices):
            picture = image_array(loader.dataset, index, cfg)
            axes[row, 0].imshow(picture)
            label = 'Query' if row == 0 else 'Cross-camera positive'
            axes[row, 0].set_title('{} | global w={:.3f}'.format(label, weights[row, 0]))
            axes[row, 0].axis('off')
            for part in range(model.lpam.queries):
                display = overlay(axes[row, part+1], picture, attention[row, part],
                                  'Part {} | w={:.3f}\nstage {}: {}'.format(part+1, weights[row,part+1],
                                      '/'.join(str(v) for v in model.patch_layers),
                                      '/'.join('{:.2f}'.format(v) for v in depths[row,part])), bound=bound)
        fig.colorbar(display, ax=axes[:, 1:].ravel().tolist(), fraction=.015, label='Attention probability')
        fig.suptitle('Epoch {} | ID {} | competitive patch pyramid {}'.format(epoch, pair['pid'], model.patch_layers))
        name = 'q{:05d}'.format(pair['query_index'])
        fig.savefig(destination/(name+'_attention.png'), dpi=140)
        plt.close(fig)
        np.savez_compressed(destination/(name+'_maps.npz'), attention=attention, depth_weights=depths,
                            attention_by_depth=parts['attention_by_depth'].float().cpu().numpy(),
                            fusion_weights=weights, patch_layers=np.asarray(model.patch_layers))
    fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
    ax.bar(range(1, 13), model.readout_weights().detach().cpu().numpy())
    ax.set(xticks=range(1, 13), xlabel='Transformer layer', ylabel='MLDC weight', ylim=(0, 1))
    fig.savefig(destination/'layer_weights.png', dpi=140)
    plt.close(fig)
    logger.info('Saved %d fixed-pair attention pictures to %s', len(pairs), destination)
