"""PFE-only training/diagnostics. The active B processor is never edited."""
import gc
import json
import logging
import time
from pathlib import Path
import torch
from utils.metrics import R1_mAP_eval
from .model import objectives
from .processor import append_csv, save_checkpoint


def save_visualizations(cfg, samples, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    folder.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        pixels = sample['image'].permute(1, 2, 0).numpy()
        pixels = (pixels * np.asarray(cfg.INPUT.PIXEL_STD) + np.asarray(cfg.INPUT.PIXEL_MEAN)).clip(0, 1)
        if 'pfe_before_norm' in sample:
            fig, axs = plt.subplots(len(cfg.MODEL.PFE_LAYERS), 4, figsize=(7, 3 * len(cfg.MODEL.PFE_LAYERS)), squeeze=False, layout='constrained')
            for row, depth in enumerate(cfg.MODEL.PFE_LAYERS):
                axs[row, 0].imshow(pixels); axs[row, 0].axis('off'); axs[row, 0].set_title(f'Input / PFE after L{depth}')
                lo = min(sample['pfe_before_norm'][row].min(), sample['pfe_after_norm'][row].min())
                hi = max(sample['pfe_before_norm'][row].max(), sample['pfe_after_norm'][row].max())
                for col, (key, title) in enumerate([('pfe_before_norm', 'Before norm'), ('pfe_after_norm', 'After norm'), ('pfe_correction_norm', 'Actual correction norm')], 1):
                    a = sample[key][row].numpy()
                    axs[row, col].imshow(pixels)
                    axs[row, col].imshow(a, cmap='turbo', alpha=.55, extent=(-.5, pixels.shape[1]-.5, pixels.shape[0]-.5, -.5), interpolation='bilinear',
                                        vmin=float(lo) if col < 3 else 0., vmax=float(hi) if col < 3 else None)
                    axs[row, col].axis('off'); axs[row, col].set_title(f'{title}\n{a.min():.3g} - {a.max():.3g}')
            fig.suptitle(f'Query {sample["index"]}, PID {sample["pid"]}; before/after share scale per row; norms are not identity contribution')
            fig.savefig(folder/f'query_{sample["index"]:05d}_pfe.png', dpi=130); plt.close(fig)
        fig, axs = plt.subplots(1, 5, figsize=(8.5, 4), layout='constrained')
        axs[0].imshow(pixels); axs[0].axis('off'); axs[0].set_title('Input')
        for col, depth in enumerate((9, 10, 11, 12), 1):
            a = sample['cls_attention'][col - 1].numpy()
            axs[col].imshow(pixels)
            axs[col].imshow(a, cmap='turbo', alpha=.55, extent=(-.5, pixels.shape[1]-.5, pixels.shape[0]-.5, -.5), interpolation='bilinear')
            axs[col].axis('off'); axs[col].set_title(f'CLS attention L{depth}\n{a.min():.3g} - {a.max():.3g}')
        fig.suptitle('CLS-to-patch softmax, mean heads; per-panel color scale, CLS self-column omitted')
        fig.savefig(folder/f'query_{sample["index"]:05d}_cls.png', dpi=130); plt.close(fig)
    (folder/'definitions.json').write_text(json.dumps(dict(pfe_layers=list(cfg.MODEL.PFE_LAYERS),
        before_after='L2 over 768 channels, shared scale before/after within each stage',
        correction='L2 of actual 768-D token increment, own scale',
        cls_attention='Mean attention heads from CLS to patch, without renormalizing after hiding CLS column',
        samples=[dict(index=s['index'], pid=s['pid']) for s in samples]), indent=2))


@torch.no_grad()
def evaluate(cfg, model, loader, nq, device, visualization_dir=None):
    model.eval()
    evaluator = R1_mAP_eval(nq, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=False)
    evaluator.reset()
    count = 0; ratios = None; entropies = None; samples = []
    for images, pids, camids, camera, view, _ in loader:
        images = images.to(device)
        data = model(images, cam_label=camera.to(device), view_label=view.to(device), diagnostics=True)
        evaluator.update((data['retrieval'], pids, camids))
        if model.enabled:
            r = data['pfe_correction_ratio'].sum(0).cpu()
            ratios = r if ratios is None else ratios + r
        attention = data['cls_attention'].flatten(2)
        conditional = attention / attention.sum(-1, keepdim=True).clamp_min(1e-12)
        import math
        entropy = (-(conditional * conditional.clamp_min(1e-30).log()).sum(-1) / math.log(conditional.shape[-1])).sum(0).cpu()
        entropies = entropy if entropies is None else entropies + entropy
        if visualization_dir is not None and cfg.TEST.PFE_VISUALIZE:
            for j in range(len(images)):
                index = count + j
                if index < nq and index in cfg.TEST.PFE_VIS_INDICES:
                    sample = dict(index=index, pid=int(pids[j]), image=images[j].detach().cpu(), cls_attention=data['cls_attention'][j].cpu())
                    if model.enabled:
                        sample.update({k: data[k][j].cpu() for k in ('pfe_before_norm', 'pfe_after_norm', 'pfe_correction_norm')})
                    samples.append(sample)
        count += len(images)
    cmc, mean_ap, distance, *_ = evaluator.compute()
    metrics = dict(mAP=float(mean_ap), rank1=float(cmc[0]), rank5=float(cmc[4]), rank10=float(cmc[9]))
    diagnostics = dict(images=count, pfe_enabled=model.enabled, pfe_layers=list(model.pfe_layers),
        mean_stage_correction_ratio=(ratios / max(count, 1)).tolist() if ratios is not None else [],
        mean_cls_patch_conditional_entropy=(entropies / max(count, 1)).tolist(),
        note='Correction/attention are diagnostics, not identity gain or restoration quality')
    if samples:
        save_visualizations(cfg, samples, Path(visualization_dir))
    del distance, evaluator
    gc.collect()
    return metrics, diagnostics


def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, nq):
    device = torch.device('cuda:0'); model.to(device)
    logger = logging.getLogger('transreid.train'); output = Path(cfg.OUTPUT_DIR)
    scaler = torch.amp.GradScaler('cuda'); best = -1.
    for epoch in range(1, cfg.SOLVER.MAX_EPOCHS + 1):
        model.train(); started = time.monotonic()
        loss_sum = id_sum = tri_sum = correct = count = updates = batches = 0
        for iteration, (images, labels, camera, view) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                data = model(images, labels, cam_label=camera.to(device), view_label=view.to(device))
                loss, identity, triplet = objectives(data, labels, cfg)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            if iteration == 1 and epoch == 1:
                norms = {prefix: sum(float(p.grad.float().square().sum()) for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None) ** .5
                         for prefix in ('pfe_adapters.', 'base.patch_embed.', 'base.blocks.', 'classifier.')}
                logger.info('PFE joint gradient norms: %s', norms)
            before = model.classifier.weight._version
            scaler.step(optimizer); scaler.update()
            updates += int(model.classifier.weight._version != before); batches += 1
            size = len(labels); count += size
            loss_sum += loss.item() * size; id_sum += identity.item() * size; tri_sum += triplet.item() * size
            correct += (data['scores'].argmax(1) == labels).sum().item()
            if iteration == 1 or iteration % cfg.SOLVER.LOG_PERIOD == 0:
                logger.info('Epoch[%d] Iter[%d/%d] Loss %.5f ID %.5f Tri %.5f Acc %.4f updates %d LR %.3g', epoch, iteration, len(train_loader), loss_sum/count, id_sum/count, tri_sum/count, correct/count, updates, optimizer.param_groups[0]['lr'])
        if count == 0:
            raise RuntimeError('Empty training loader')
        scheduler.step(epoch)
        row = dict(epoch=epoch, loss=loss_sum/count, identity=id_sum/count, triplet=tri_sum/count,
                   accuracy=correct/count, batches=batches, optimizer_updates=updates, amp_skipped_steps=batches-updates, seconds=time.monotonic()-started)
        append_csv(output/'training.csv', row)
        append_csv(output/'layer_weights.csv', dict(epoch=epoch, **{f'layer_{i+1}': float(w) for i, w in enumerate(model.readout_weights())}))
        logger.info('Epoch %d complete: %s', epoch, row)
        metrics = None
        if epoch % cfg.SOLVER.EVAL_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS:
            metrics, diagnostics = evaluate(cfg, model, val_loader, nq, device, output/'visualizations'/f'epoch_{epoch:03d}')
            append_csv(output/'evaluation.csv', dict(epoch=epoch, **metrics))
            with (output/'evaluation_diagnostics.jsonl').open('a') as f:
                f.write(json.dumps(dict(epoch=epoch, metrics=metrics, **diagnostics))+'\n')
            logger.info('PFE validation epoch %d: %s; %s', epoch, metrics, diagnostics)
            if metrics['mAP'] > best:
                best = metrics['mAP']; save_checkpoint(output/'transformer_best.pth', model, cfg, epoch, metrics)
        if epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS:
            save_checkpoint(output/f'transformer_{epoch}.pth', model, cfg, epoch, metrics)
