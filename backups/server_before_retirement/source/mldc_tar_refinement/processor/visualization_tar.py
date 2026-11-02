"""Fixed-pair TAR maps; no additional ranking or attribution backward passes."""
import json
from pathlib import Path
import numpy as np
import torch
from .visualization import fixed_pairs, plt_module, image_batch, image_array, overlay


@torch.no_grad()
def save_visualizations(cfg, model, loader, num_query, epoch, logger):
    was_training = model.training
    model.eval()
    try:
        output = Path(cfg.OUTPUT_DIR)
        destination = output / 'visualizations' / 'epoch_{:03d}'.format(epoch)
        destination.mkdir(parents=True, exist_ok=True)
        pairs = fixed_pairs(loader.dataset.dataset, num_query, cfg.TEST.MLDC_VIS_SAMPLES, cfg.SOLVER.SEED)
        pair_path = output / 'fixed_visualization_pairs.json'
        if pair_path.exists() and json.loads(pair_path.read_text(encoding='utf-8')) != pairs:
            raise ValueError('Fixed visualization pairs changed')
        pair_path.write_text(json.dumps(pairs, indent=2), encoding='utf-8')
        device = next(model.parameters()).device
        plt = plt_module()
        for pair in pairs:
            indices = [pair['query_index'], pair['reference_index']]
            images, camera, view = image_batch(loader.dataset, indices, device)
            parts = model(images, cam_label=camera, view_label=view, return_components=True)
            grid = parts['grid_hw']
            norms = parts['tar_patch_norms'].cpu().numpy().reshape(2, *grid)
            weights = parts['tar_weights'].cpu().numpy().reshape(2, *grid)
            mask = parts['tar_mask'].float().cpu().numpy().reshape(2, *grid)
            attention = parts['tar_attention'].cpu().numpy().reshape(2, *grid)
            gate = parts['tar_gate'].cpu().numpy().reshape(2)
            ratio = (parts['tar_delta'].norm(dim=1) /
                     (parts['raw_readout']+parts['raw_last_cls']).norm(dim=1).clamp_min(1e-12)).cpu().numpy()
            fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
            for row, index in enumerate(indices):
                picture = image_array(loader.dataset, index, cfg)
                axes[row, 0].imshow(picture)
                axes[row, 0].axis('off')
                axes[row, 0].set_title('{} | gate={:.3f}\ndelta/base norm={:.3f}'.format(
                    'Query' if row == 0 else 'Cross-camera positive', gate[row], ratio[row]), fontsize=9)
                for col, values, title, bound in [
                    (1, norms, 'Raw patch L2 norm', float(norms.max())),
                    (2, weights, 'TRM weight (not foreground)', 1.0),
                    (3, mask, 'Candidate mask', 1.0),
                    (4, attention, 'CATS aggregation probability', float(attention.max())),
                ]:
                    overlay(axes[row, col], picture, values[row], title, bound=bound)
            fig.suptitle('Epoch {} | ID {} | TAR on raw block-12 tokens; maps are not retrieval contributions'.format(epoch, pair['pid']))
            name = 'q{:05d}'.format(pair['query_index'])
            fig.savefig(destination / (name+'_tar.png'), dpi=140)
            plt.close(fig)
            np.savez_compressed(destination / (name+'_maps.npz'), patch_norms=norms,
                                trm_weights=weights, candidate_mask=mask, cats_attention=attention,
                                tar_gate=gate, tar_delta_base_norm_ratio=ratio, patch_grid=np.asarray(grid),
                                mldc_weights=model.readout_weights().detach().cpu().numpy())
        fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
        ax.bar(range(1, 13), model.readout_weights().detach().cpu().numpy())
        ax.set(xticks=range(1, 13), xlabel='Transformer layer', ylabel='MLDC weight', ylim=(0, 1))
        fig.savefig(destination/'layer_weights.png', dpi=140)
        plt.close(fig)
        logger.info('Saved %d fixed-pair TAR maps to %s', len(pairs), destination)
    finally:
        model.train(was_training)
