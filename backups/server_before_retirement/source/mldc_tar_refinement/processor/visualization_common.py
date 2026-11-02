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


