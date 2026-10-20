"""True query attention, signed retrieval-margin attribution, and rank evidence."""
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


def render_ranks(cfg, dataset, cache, query, destination, label):
    plt = plt_module()
    fig, axes = plt.subplots(3, 11, figsize=(16, 7), constrained_layout=True)
    pid = cache['pids'][query]
    for row, (name, title) in enumerate([('mldc_after_bn', 'Global'), ('lpam_after_bn', 'Local'), ('retrieval', 'Joint')]):
        rank = cache['rankings'][name][query]
        indices = [query]+rank['top10']
        for col, axis in enumerate(axes[row]):
            if col >= len(indices):
                axis.axis('off')
                continue
            i = indices[col]
            axis.imshow(image_array(dataset, i, cfg))
            color = 'black' if col == 0 else '#118642' if cache['pids'][i] == pid else '#cb3131'
            for spine in axis.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            axis.set_xticks([]); axis.set_yticks([])
            axis.set_title('Query' if col == 0 else '#{} | ID {}'.format(col, cache['pids'][i]), fontsize=8)
        axes[row, 0].set_ylabel('{}\nAP {:.3f}'.format(title, rank['AP'] or 0), fontsize=10)
    fig.suptitle('{} | query {} | green = same ID; same ID/camera excluded'.format(label, query), fontsize=12)
    fig.savefig(destination/'q{:05d}_ranks.png'.format(query), dpi=140)
    plt.close(fig)


def margin_evidence(cfg, model, dataset, cache, pair, destination, label, device):
    query, positive = pair['query_index'], pair['reference_index']
    global_features = F.normalize(cache['features']['mldc_after_bn'], dim=1)
    similarities = global_features[query] @ global_features[cache['num_query']:].T
    negative_mask = cache['pids'][cache['num_query']:] != cache['pids'][query]
    if not negative_mask.any():
        raise ValueError('Visualization needs a different-ID gallery negative')
    similarities[~torch.as_tensor(negative_mask)] = -torch.inf
    negative = int(similarities.argmax())+cache['num_query']
    images, camera, view = image_batch(dataset, [positive, negative], device)
    with torch.no_grad():
        reference = model(images, cam_label=camera, view_label=view, return_components=True)
    captured = {}
    def capture(_module, _inputs, output):
        # Frozen backbone still needs a differentiable token input for attribution.
        tokens = output.detach().requires_grad_(True)
        captured['patches'] = tokens
        return tokens
    hook = model.base.patch_embed.register_forward_hook(capture)
    try:
        images, camera, view = image_batch(dataset, [query], device)
        with torch.enable_grad():
            query_parts = model(images, cam_label=camera, view_label=view, return_components=True)
            grid = query_parts['grid_hw']
            attribution, margins = {}, {}
            for name in ('mldc_after_bn', 'lpam_after_bn', 'retrieval'):
                q = F.normalize(query_parts[name].float(), dim=1)
                r = F.normalize(reference[name].detach().float(), dim=1)
                margin = (q*(r[0:1]-r[1:2])).sum()
                gradient = torch.autograd.grad(margin, captured['patches'], retain_graph=True)[0]
                attribution[name] = (captured['patches'].float()*gradient.float()).sum(-1).reshape(*grid).detach().cpu().numpy()
                margins[name] = float(margin.detach())
            qparts = F.normalize(query_parts['bn_parts'].float(), dim=-1)
            rparts = F.normalize(reference['bn_parts'].float(), dim=-1)
            part_margins = (qparts*(rparts[0:1]-rparts[1:2])).sum(-1)[0].detach().cpu().numpy()
            contributions = part_margins*model.local_weight/model.lpam.queries
            global_contribution = (1-model.local_weight)*margins['mldc_after_bn']
            if not np.isclose(margins['retrieval'], global_contribution+contributions.sum(), atol=2e-5):
                raise AssertionError('Joint similarity decomposition failed')
            attention = query_parts['attention'][0].detach().cpu().numpy().reshape(model.lpam.queries, *grid)
            positive_attention = reference['attention'][0].detach().cpu().numpy().reshape(model.lpam.queries, *grid)
            depths = query_parts['depth_weights'][0].detach().cpu().numpy()
            by_depth = query_parts['attention_by_depth'][0].detach().cpu().numpy()
    finally:
        hook.remove()
    plt = plt_module()
    fig, axes = plt.subplots(4, 4, figsize=(12, 12), constrained_layout=True)
    for axis, index, title in zip(axes[0, :3], [query, positive, negative], ['Query', 'Cross-camera positive', 'Global hard negative']):
        axis.imshow(image_array(dataset, index, cfg)); axis.set_title(title, fontsize=10); axis.axis('off')
    ax = axes[0, 3]
    values = np.r_[global_contribution, contributions]
    ax.bar(['G', 'P1', 'P2', 'P3', 'P4'], values, color=['#347caa']+['#d7934d']*4)
    ax.axhline(0, color='black', linewidth=.6)
    ax.set_title('Exact weighted margin contributions\nJoint margin = {:.4f}'.format(margins['retrieval']), fontsize=9)
    ax.set_ylabel('cos(query, pos) - cos(query, neg)', fontsize=8)
    picture = image_array(dataset, query, cfg)
    positive_picture = image_array(dataset, positive, cfg)
    attention_bound = max(attention.max(), positive_attention.max())
    for i in range(model.lpam.queries):
        display = overlay(axes[1, i], picture, attention[i], 'Query | Part {}'.format(i+1), bound=attention_bound)
        overlay(axes[2, i], positive_picture, positive_attention[i], 'Positive | Part {}'.format(i+1), bound=attention_bound)
    fig.colorbar(display, ax=axes[1:3].ravel().tolist(), fraction=.015, label='Attention probability (shared scale)')
    signed_bound = max(np.abs(values).max() for values in attribution.values())
    for i, (name, title) in enumerate([('mldc_after_bn', 'Global'), ('lpam_after_bn', 'Local'), ('retrieval', 'Joint')]):
        display = overlay(axes[3, i], picture, attribution[name], '{} signed margin attribution'.format(title),
                          signed=True, bound=signed_bound)
    fig.colorbar(display, ax=axes[3, :3].tolist(), fraction=.02, label='Signed gradient x patch embedding')
    display = axes[3, 3].imshow(depths, vmin=0, vmax=1, cmap='viridis', aspect='auto')
    axes[3, 3].set(xticks=range(len(model.patch_layers)), xticklabels=model.patch_layers,
                  yticks=range(model.lpam.queries), yticklabels=['P1', 'P2', 'P3', 'P4'],
                  xlabel='Transformer layer', title='Input-dependent depth gates')
    for i in range(model.lpam.queries):
        for j in range(len(model.patch_layers)):
            axes[3, 3].text(j, i, '{:.2f}'.format(depths[i, j]), ha='center', va='center', color='white', fontsize=8)
    fig.suptitle('{} | q{} / p{} / n{} | red attribution supports positive over negative'.format(label, query, positive, negative), fontsize=11)
    fig.savefig(destination/'q{:05d}_attention_margin.png'.format(query), dpi=145)
    plt.close(fig)
    np.savez_compressed(destination/'q{:05d}_maps.npz'.format(query), attention=attention,
                        positive_attention=positive_attention, attention_by_depth=by_depth,
                        depth_weights=depths, **{'signed_'+k: v for k, v in attribution.items()})
    evidence = {**pair, 'negative_index': negative, 'negative_path': str(dataset.dataset[negative][0]),
                'label': label, 'margins': margins, 'weighted_global_margin': global_contribution,
                'weighted_part_margins': contributions.tolist(), 'depth_weights': depths.tolist(),
                'note': 'Attention and gradient x embedding are diagnostics. Complementarity is measured by retrieval changes.'}
    (destination/'q{:05d}_evidence.json'.format(query)).write_text(json.dumps(evidence, indent=2), encoding='utf-8')


def save_visualizations(cfg, model, loader, num_query, epoch, logger, cache):
    model.eval()
    output = Path(cfg.OUTPUT_DIR)
    destination = output/'visualizations'/'epoch_{:03d}'.format(epoch)
    destination.mkdir(parents=True, exist_ok=True)
    records = loader.dataset.dataset
    manifest = output/'fixed_visualization_pairs.json'
    pairs = fixed_pairs(records, num_query, cfg.TEST.MLDC_VIS_SAMPLES, cfg.SOLVER.SEED)
    if manifest.exists():
        stored = json.loads(manifest.read_text(encoding='utf-8'))
        if stored != pairs:
            raise ValueError('Fixed visualization sample order changed')
    else:
        manifest.write_text(json.dumps(pairs, indent=2), encoding='utf-8')
    selected = [(pair, 'Fixed cross-camera pair') for pair in pairs]
    used = {pair['query_index'] for pair in pairs}
    for status in ('repair', 'harm', 'both_wrong'):
        if cfg.TEST.LPAM_VIS_CASES_PER_GROUP <= 0:
            continue
        count = 0
        for row in cache['query_changes']:
            query = row['query_index']
            if row['status'] != status or query in used:
                continue
            positives = [i for i in range(num_query, len(records))
                         if records[i][1] == records[query][1] and records[i][2] != records[query][2]]
            if not positives:
                continue
            selected.append(({'query_index': query, 'reference_index': positives[0],
                              'query_path': str(records[query][0]), 'reference_path': str(records[positives[0]][0])}, status))
            used.add(query)
            count += 1
            if count >= cfg.TEST.LPAM_VIS_CASES_PER_GROUP:
                break
    device = next(model.parameters()).device
    for pair, label in selected:
        margin_evidence(cfg, model, loader.dataset, cache, pair, destination, label, device)
        render_ranks(cfg, loader.dataset, cache, pair['query_index'], destination, label)
    plt = plt_module()
    fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
    ax.bar(range(1, 13), model.readout_weights().detach().cpu().numpy(), color='#347caa')
    ax.set(xticks=range(1, 13), xlabel='Transformer layer', ylabel='MLDC weight', ylim=(0, 1))
    fig.savefig(destination/'layer_weights.png', dpi=150)
    plt.close(fig)
    logger.info('Saved %d attention/margin/rank cases to %s', len(selected), destination)
