"""PFE on/off effect in a fixed network, with shared-scale spatial overlays."""
import json
from pathlib import Path
import torch


@torch.no_grad()
def probe_effect(model, images, cameras=None, views=None):
    training, enabled = model.training, model.enabled
    try:
        model.eval(); model.enabled = False
        before = model(images, cam_label=cameras, view_label=views, diagnostics=True)
        model.enabled = enabled
        after = model(images, cam_label=cameras, view_label=views, diagnostics=True)
        def relative(a, b):
            return ((a.float()-b.float()).flatten(1).norm(dim=1) /
                    b.float().flatten(1).norm(dim=1).clamp_min(1e-12)).cpu()
        summary = dict(images=len(images),
            stage_ratio_mean=after['pfe_correction_ratio'].mean(0).cpu().tolist() if enabled else [],
            descriptor_relative_change_mean=float(relative(after['retrieval'], before['retrieval']).mean()),
            cls_relative_change_mean=[float(relative(after['layers'][i], before['layers'][i]).mean()) for i in (8,9,10,11)],
            cls_attention_abs_change_mean=float((after['cls_attention']-before['cls_attention']).abs().mean()),
            note='On/off within the same network; not a retrieval accuracy measurement')
        maps = dict(images=images.detach().cpu(), attention_before=before['cls_attention'].cpu(),
                    attention_after=after['cls_attention'].cpu())
        for key in ('pfe_before_norm', 'pfe_after_norm', 'pfe_correction_norm'):
            if key in after: maps[key] = after[key].cpu()
        return summary, maps
    finally:
        model.enabled = enabled; model.train(training)


def render_probe(cfg, summary, maps, folder, prefix='probe'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    for i in range(min(3, len(maps['images']))):
        pixels = maps['images'][i].permute(1,2,0).numpy()
        pixels = (pixels*np.asarray(cfg.INPUT.PIXEL_STD)+np.asarray(cfg.INPUT.PIXEL_MEAN)).clip(0,1)
        extent = (-.5, pixels.shape[1]-.5, pixels.shape[0]-.5, -.5)
        fig, axes = plt.subplots(4, 7, figsize=(13, 11), layout='constrained')
        for row, (stage, depth) in enumerate(zip(cfg.MODEL.PFE_LAYERS, (9,10,11,12))):
            before = maps['pfe_before_norm'][i,row].numpy()
            after = maps['pfe_after_norm'][i,row].numpy()
            correction = maps['pfe_correction_norm'][i,row].numpy()
            a0 = maps['attention_before'][i,row].numpy()
            a1 = maps['attention_after'][i,row].numpy()
            diff = a1-a0
            for axis in axes[row]: axis.imshow(pixels); axis.axis('off')
            axes[row,0].set_title(f'Input / PFE L{stage}')
            lo, hi = min(before.min(),after.min()), max(before.max(),after.max())
            for col, data, title in ((1,before,'Patch before'), (2,after,'Patch after')):
                axes[row,col].imshow(data,cmap='turbo',vmin=lo,vmax=hi,alpha=.6,extent=extent,interpolation='bilinear')
                axes[row,col].set_title(title)
            axes[row,3].imshow(correction,cmap='turbo',vmin=0,alpha=.6,extent=extent,interpolation='bilinear')
            axes[row,3].set_title(f'Actual delta L{stage}\nmax={correction.max():.3g}')
            lo, hi = min(a0.min(),a1.min()), max(a0.max(),a1.max())
            for col, data, title in ((4,a0,'PFE off'), (5,a1,'PFE on')):
                axes[row,col].imshow(data,cmap='turbo',vmin=lo,vmax=hi,alpha=.6,extent=extent,interpolation='bilinear')
                axes[row,col].set_title(f'CLS L{depth} / {title}')
            limit=max(float(np.abs(diff).max()),1e-12)
            axes[row,6].imshow(diff,cmap='coolwarm',vmin=-limit,vmax=limit,alpha=.65,extent=extent,interpolation='bilinear')
            axes[row,6].set_title(f'Attention on-off\nmax abs={limit:.2g}')
        fig.suptitle('Paired panels share color scales; delta panel uses its own symmetric scale.\n'
            'Rows pair two separate diagnostics: PFE insertion stage / CLS readout layer.')
        fig.savefig(folder/f'{prefix}_{i:02d}.png',dpi=120); plt.close(fig)
    (folder/f'{prefix}.json').write_text(json.dumps(summary,indent=2))
