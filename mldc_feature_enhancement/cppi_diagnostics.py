"""CPPI functional probes and overlays; supports any configured injection count."""
import json
from pathlib import Path
import torch


def query_statistics(distance,pids,camids,nq):
    """Per-query AP/Rank-1 using the repository's same-ID/same-camera filter."""
    import numpy as np
    pids=np.asarray(pids);camids=np.asarray(camids)
    gallery_pids=pids[nq:];gallery_cams=camids[nq:]
    rows=[]
    for i,distances in enumerate(distance):
        order=np.argsort(distances)
        keep=~((gallery_pids[order]==pids[i])&(gallery_cams[order]==camids[i]))
        order=order[keep];matches=gallery_pids[order]==pids[i]
        valid=bool(matches.any())
        ap=float(((matches.cumsum()/np.arange(1,len(matches)+1))*matches).sum()/matches.sum()) if valid else None
        rows.append(dict(index=i,pid=int(pids[i]),camera=int(camids[i]),valid=valid,
            rank1=bool(matches[0]) if valid else None,ap=ap,
            top_gallery_index=int(order[0]) if len(order) else None))
    return rows


def summarize_query_changes(on,off):
    if [(r['index'],r['pid'],r['camera'],r['valid']) for r in on]!=[(r['index'],r['pid'],r['camera'],r['valid']) for r in off]:
        raise ValueError('On/off evaluations must use identical queries and gallery validity')
    pairs=[(a,b) for a,b in zip(on,off) if a['valid']]
    if not pairs:raise ValueError('No valid queries in on/off comparison')
    corrected=[a['index'] for a,b in pairs if a['rank1'] and not b['rank1']]
    broken=[a['index'] for a,b in pairs if b['rank1'] and not a['rank1']]
    return dict(valid_queries=len(pairs),corrected_count=len(corrected),broken_count=len(broken),
        net_corrected=len(corrected)-len(broken),corrected_query_indices=corrected,broken_query_indices=broken,
        mean_ap_delta=sum(a['ap']-b['ap'] for a,b in pairs)/len(pairs),
        ap_improved_queries=sum(a['ap']>b['ap']+1e-12 for a,b in pairs),
        ap_worsened_queries=sum(a['ap']<b['ap']-1e-12 for a,b in pairs),
        note='Inference on/off in one checkpoint, not an independently trained baseline')


@torch.no_grad()
def probe_effect(model,images,cameras=None,views=None):
    training,enabled=model.training,model.enabled
    try:
        model.eval();model.enabled=False
        before=model(images,cam_label=cameras,view_label=views,diagnostics=True)
        model.enabled=enabled
        after=model(images,cam_label=cameras,view_label=views,diagnostics=True)
        def relative(a,b):
            return ((a.float()-b.float()).flatten(1).norm(dim=1)/b.float().flatten(1).norm(dim=1).clamp_min(1e-12)).cpu()
        summary=dict(images=len(images),cppi_enabled=enabled,
            stage_budgets=model.correction_budgets(),
            stage_ratio_mean=after['cppi_correction_ratio'].mean(0).cpu().tolist() if enabled else [],
            scale_mass_mean=after['cppi_scale_mass'].mean(0).cpu().tolist() if enabled else [],
            descriptor_relative_change_mean=float(relative(after['retrieval'],before['retrieval']).mean()),
            cls_relative_change_mean=[float(relative(after['layers'][i],before['layers'][i]).mean()) for i in (8,9,10,11)],
            cls_attention_abs_change_mean=float((after['cls_attention']-before['cls_attention']).abs().mean()),
            note='Same-network on/off probe, not retrieval accuracy or an independently trained baseline')
        maps=dict(images=images.detach().cpu(),attention_before=before['cls_attention'].cpu(),attention_after=after['cls_attention'].cpu())
        for key in ('cppi_before_norm','cppi_after_norm','cppi_correction_norm'):
            if key in after:maps[key]=after[key].cpu()
        return summary,maps
    finally:
        model.enabled=enabled;model.train(training)


def render_probe(cfg,summary,maps,folder,prefix='probe'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for i in range(min(3,len(maps['images']))):
        pixels=maps['images'][i].permute(1,2,0).numpy()
        pixels=(pixels*np.asarray(cfg.INPUT.PIXEL_STD)+np.asarray(cfg.INPUT.PIXEL_MEAN)).clip(0,1)
        extent=(-.5,pixels.shape[1]-.5,pixels.shape[0]-.5,-.5)
        def overlay(ax,a,cmap='turbo',vmin=None,vmax=None):
            ax.imshow(pixels);ax.imshow(a,cmap=cmap,alpha=.58,extent=extent,interpolation='bilinear',vmin=vmin,vmax=vmax);ax.axis('off')
        if 'cppi_before_norm' in maps:
            fig,axes=plt.subplots(len(cfg.MODEL.CPPI_LAYERS),4,figsize=(8,3*len(cfg.MODEL.CPPI_LAYERS)),squeeze=False,layout='constrained')
            for row,depth in enumerate(cfg.MODEL.CPPI_LAYERS):
                axes[row,0].imshow(pixels);axes[row,0].axis('off');axes[row,0].set_title(f'Input / after L{depth}')
                a=maps['cppi_before_norm'][i,row].numpy();b=maps['cppi_after_norm'][i,row].numpy()
                lo,hi=min(a.min(),b.min()),max(a.max(),b.max())
                for col,z,title in ((1,a,'Patch before'),(2,b,'Patch after')):
                    overlay(axes[row,col],z,vmin=lo,vmax=hi);axes[row,col].set_title(title)
                delta=maps['cppi_correction_norm'][i,row].numpy()
                overlay(axes[row,3],delta,vmin=0);axes[row,3].set_title(f'Actual CPPI correction\nmax={delta.max():.3g} / budget={summary["stage_budgets"][str(depth)]*100:.2f}%')
            fig.suptitle('Paired patch norms share scale; correction uses its own scale. Norms are not identity attribution.')
            fig.savefig(folder/f'{prefix}_{i:02d}_patch.png',dpi=120);plt.close(fig)
        fig,axes=plt.subplots(4,4,figsize=(8,11),layout='constrained')
        for row,depth in enumerate((9,10,11,12)):
            axes[row,0].imshow(pixels);axes[row,0].axis('off');axes[row,0].set_title(f'Input / CLS L{depth}')
            a=maps['attention_before'][i,row].numpy();b=maps['attention_after'][i,row].numpy()
            lo,hi=min(a.min(),b.min()),max(a.max(),b.max())
            for col,z,title in ((1,a,'CPPI off'),(2,b,'CPPI on')):
                overlay(axes[row,col],z,vmin=lo,vmax=hi);axes[row,col].set_title(title)
            diff=b-a;limit=max(float(abs(diff).max()),1e-12)
            overlay(axes[row,3],diff,'coolwarm',-limit,limit);axes[row,3].set_title(f'On-off\nmax abs={limit:.3g}')
        fig.suptitle('Mean-head CLS-to-patch softmax; on/off share scale; difference uses its own symmetric scale.')
        fig.savefig(folder/f'{prefix}_{i:02d}_cls.png',dpi=120);plt.close(fig)
    (folder/f'{prefix}.json').write_text(json.dumps(summary,indent=2))
