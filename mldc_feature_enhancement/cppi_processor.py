"""CPPI-only training/diagnostics. The active B processor is never edited."""
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
        if 'cppi_before_norm' in sample:
            fig, axs = plt.subplots(len(cfg.MODEL.CPPI_LAYERS), 4, figsize=(7, 3 * len(cfg.MODEL.CPPI_LAYERS)), squeeze=False, layout='constrained')
            for row, depth in enumerate(cfg.MODEL.CPPI_LAYERS):
                axs[row, 0].imshow(pixels); axs[row, 0].axis('off'); axs[row, 0].set_title(f'Input / CPPI after L{depth}')
                lo = min(sample['cppi_before_norm'][row].min(), sample['cppi_after_norm'][row].min())
                hi = max(sample['cppi_before_norm'][row].max(), sample['cppi_after_norm'][row].max())
                for col, (key, title) in enumerate([('cppi_before_norm', 'Before norm'), ('cppi_after_norm', 'After norm'), ('cppi_correction_norm', 'Actual correction norm')], 1):
                    a = sample[key][row].numpy()
                    axs[row, col].imshow(pixels)
                    axs[row, col].imshow(a, cmap='turbo', alpha=.55, extent=(-.5, pixels.shape[1]-.5, pixels.shape[0]-.5, -.5), interpolation='bilinear',
                                        vmin=float(lo) if col < 3 else 0., vmax=float(hi) if col < 3 else None)
                    axs[row, col].axis('off'); axs[row, col].set_title(f'{title}\n{a.min():.3g} - {a.max():.3g}' + (f' / budget={float(sample["cppi_budget"][row])*100:.2f}%' if col == 3 else ''))
            fig.suptitle(f'Query {sample["index"]}, PID {sample["pid"]}; before/after share scale per row; norms are not identity contribution')
            fig.savefig(folder/f'query_{sample["index"]:05d}_cppi.png', dpi=130); plt.close(fig)
        fig, axs = plt.subplots(1, 5, figsize=(8.5, 4), layout='constrained')
        axs[0].imshow(pixels); axs[0].axis('off'); axs[0].set_title('Input')
        for col, depth in enumerate((9, 10, 11, 12), 1):
            a = sample['cls_attention'][col - 1].numpy()
            axs[col].imshow(pixels)
            axs[col].imshow(a, cmap='turbo', alpha=.55, extent=(-.5, pixels.shape[1]-.5, pixels.shape[0]-.5, -.5), interpolation='bilinear')
            axs[col].axis('off'); axs[col].set_title(f'CLS attention L{depth}\n{a.min():.3g} - {a.max():.3g}')
        fig.suptitle('CLS-to-patch softmax, mean heads; per-panel color scale, CLS self-column omitted')
        fig.savefig(folder/f'query_{sample["index"]:05d}_cls.png', dpi=130); plt.close(fig)
    (folder/'definitions.json').write_text(json.dumps(dict(cppi_layers=list(cfg.MODEL.CPPI_LAYERS),
        before_after='L2 over 768 channels, shared scale before/after within each stage',
        correction='L2 of actual 768-D token increment, own scale',
        cls_attention='Mean attention heads from CLS to patch, without renormalizing after hiding CLS column',
        samples=[dict(index=s['index'], pid=s['pid']) for s in samples]), indent=2))


@torch.no_grad()
def evaluate(cfg, model, loader, nq, device, visualization_dir=None, query_details=False):
    model.eval()
    evaluator = R1_mAP_eval(nq, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=False)
    evaluator.reset()
    count = 0; ratios = None; entropies = None; scale_mass = None; samples = []
    for images, pids, camids, camera, view, _ in loader:
        images = images.to(device)
        data = model(images, cam_label=camera.to(device), view_label=view.to(device), diagnostics=True)
        evaluator.update((data['retrieval'], pids, camids))
        if model.enabled:
            r = data['cppi_correction_ratio'].sum(0).cpu()
            ratios = r if ratios is None else ratios + r
            s = data['cppi_scale_mass'].sum(0).cpu()
            scale_mass = s if scale_mass is None else scale_mass + s
        attention = data['cls_attention'].flatten(2)
        conditional = attention / attention.sum(-1, keepdim=True).clamp_min(1e-12)
        import math
        entropy = (-(conditional * conditional.clamp_min(1e-30).log()).sum(-1) / math.log(conditional.shape[-1])).sum(0).cpu()
        entropies = entropy if entropies is None else entropies + entropy
        if visualization_dir is not None and cfg.TEST.CPPI_VISUALIZE:
            for j in range(len(images)):
                index = count + j
                if index < nq and index in cfg.TEST.CPPI_VIS_INDICES:
                    sample = dict(index=index, pid=int(pids[j]), image=images[j].detach().cpu(), cls_attention=data['cls_attention'][j].cpu())
                    if model.enabled:
                        sample.update({k: data[k][j].cpu() for k in ('cppi_before_norm', 'cppi_after_norm', 'cppi_correction_norm', 'cppi_budget')})
                    samples.append(sample)
        count += len(images)
    cmc, mean_ap, distance, *_ = evaluator.compute()
    metrics = dict(mAP=float(mean_ap), rank1=float(cmc[0]), rank5=float(cmc[4]), rank10=float(cmc[9]))
    diagnostics = dict(stage_budgets=model.correction_budgets(), images=count, cppi_enabled=model.enabled, cppi_layers=list(model.cppi_layers),
        mean_stage_correction_ratio=(ratios / max(count, 1)).tolist() if ratios is not None else [],
        mean_scale_mass=(scale_mass / max(count, 1)).tolist() if scale_mass is not None else [],
        mean_cls_patch_conditional_entropy=(entropies / max(count, 1)).tolist(),
        note='Correction/attention are diagnostics, not identity gain or restoration quality')
    if query_details:
        from .cppi_diagnostics import query_statistics
        diagnostics['query_statistics'] = query_statistics(distance, evaluator.pids, evaluator.camids, nq)
    if samples:
        save_visualizations(cfg, samples, Path(visualization_dir))
    del distance, evaluator
    gc.collect()
    return metrics, diagnostics


from .cppi_optim import lr_summary
from .cppi_diagnostics import probe_effect, render_probe

def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, nq, probe_loader=None):
    device = torch.device('cuda:0'); model.to(device)
    logger = logging.getLogger('transreid.train'); output = Path(cfg.OUTPUT_DIR)
    scaler = torch.amp.GradScaler('cuda'); best = -1.
    fixed_probe = None
    for epoch in range(1, cfg.SOLVER.MAX_EPOCHS + 1):
        model.train(); started = time.monotonic()
        loss_sum = id_sum = tri_sum = correct = count = updates = batches = 0
        for iteration, (images, labels, camera, view) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            if fixed_probe is None:
                if probe_loader is None:
                    probe_images, probe_cam, probe_view = images[:6].detach().cpu(), camera[:6].cpu(), view[:6].cpu()
                else:
                    probe_images, _, _, probe_cam, probe_view, _ = next(iter(probe_loader))
                fixed_probe = (probe_images, probe_cam, probe_view)
                summary, maps = probe_effect(model, probe_images.to(device), probe_cam.to(device), probe_view.to(device))
                summary.update(epoch=0, kind='startup_training_images_no_optimizer_step')
                render_probe(cfg, summary, maps, output/'effect_probes'/'epoch_000')
                with (output/'effect_probes.jsonl').open('a') as f: f.write(json.dumps(summary)+'\n')
                logger.info('CPPI startup actual effect: %s', summary)
                del summary, maps
            inspect = iteration == 1
            with torch.autocast('cuda'):
                data = model(images, labels, cam_label=camera.to(device), view_label=view.to(device), diagnostics=inspect)
                loss, identity, triplet = objectives(data, labels, cfg)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            if inspect:
                snapshots = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith('cppi_')}
                detail = dict(epoch=epoch, lr=lr_summary(optimizer),
                    stage_ratio_mean=data['cppi_correction_ratio'].mean(0).cpu().tolist() if model.enabled else [],
                    scale_mass_mean=data['cppi_scale_mass'].mean(0).cpu().tolist() if model.enabled else [])
                detail['stage_budget_before_step'] = data['cppi_budget'].mean(0).cpu().tolist() if model.enabled else []
                detail['budget_logit_gradients'] = {depth:float(module.budget_logit.grad) for depth,module in model.cppi_injectors.items() if module.budget_logit.grad is not None}
                detail['gradient_norms'] = {part: sum(float(p.grad.float().square().sum()) for n,p in model.named_parameters()
                    if n.startswith('cppi_') and part in n and p.grad is not None)**.5
                    for part in ('cppi_pyramid.stem.', 'cppi_pyramid.stage4.', 'cppi_pyramid.stage5.', 'cppi_pyramid.mrfp.', 'cppi_pyramid.project.', '.query', '.key.', '.value.', '.up.')}
            if iteration == 1 and epoch == 1:
                norms = {prefix: sum(float(p.grad.float().square().sum()) for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None) ** .5
                         for prefix in ('cppi_pyramid.', 'cppi_injectors.', 'base.patch_embed.', 'base.blocks.', 'classifier.')}
                logger.info('CPPI joint gradient norms: %s', norms)
            before = model.classifier.weight._version
            scaler.step(optimizer); scaler.update()
            updates += int(model.classifier.weight._version != before); batches += 1
            if inspect:
                detail['optimizer_step_applied'] = model.classifier.weight._version != before
                detail['stage_budget_after_step'] = model.correction_budgets()
                detail['relative_updates'] = {}
                for part in ('cppi_pyramid.stem.', 'cppi_pyramid.stage4.', 'cppi_pyramid.stage5.', 'cppi_pyramid.mrfp.', 'cppi_pyramid.project.', '.query', '.key.', '.value.', '.up.'):
                    old_sq = change_sq = direction_sq = 0.
                    for name, parameter in model.named_parameters():
                        if name not in snapshots or part not in name: continue
                        old = snapshots[name].float(); new = parameter.detach().float()
                        change = new-old
                        scale = (new*old).sum()/old.square().sum().clamp_min(1e-30)
                        old_sq += float(old.square().sum()); change_sq += float(change.square().sum())
                        direction_sq += float((new-scale*old).square().sum())
                    detail['relative_updates'][part] = dict(total=(change_sq/max(old_sq,1e-30))**.5,
                        direction_after_removing_scalar_rescale=(direction_sq/max(old_sq,1e-30))**.5)
                with (output/'cppi_learning_diagnostics.jsonl').open('a') as f: f.write(json.dumps(detail)+'\n')
                logger.info('CPPI magnitude / LR / actual parameter update: %s', detail)
                del snapshots, detail
            size = len(labels); count += size
            loss_sum += loss.item() * size; id_sum += identity.item() * size; tri_sum += triplet.item() * size
            correct += (data['scores'].argmax(1) == labels).sum().item()
            if iteration == 1 or iteration % cfg.SOLVER.LOG_PERIOD == 0:
                logger.info('Epoch[%d] Iter[%d/%d] Loss %.5f ID %.5f Tri %.5f Acc %.4f updates %d LR %.3g', epoch, iteration, len(train_loader), loss_sum/count, id_sum/count, tri_sum/count, correct/count, updates, optimizer.param_groups[0]['lr'])
        if count == 0:
            raise RuntimeError('Empty training loader')
        scheduler.step()
        row = dict(epoch=epoch, loss=loss_sum/count, identity=id_sum/count, triplet=tri_sum/count,
                   accuracy=correct/count, batches=batches, optimizer_updates=updates, amp_skipped_steps=batches-updates, seconds=time.monotonic()-started)
        append_csv(output/'training.csv', row)
        append_csv(output/'layer_weights.csv', dict(epoch=epoch, **{f'layer_{i+1}': float(w) for i, w in enumerate(model.readout_weights())}))
        logger.info('Epoch %d complete: %s', epoch, row)
        metrics = None
        if epoch % cfg.SOLVER.EVAL_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS:
            metrics, diagnostics = evaluate(cfg, model, val_loader, nq, device, output/'visualizations'/f'epoch_{epoch:03d}')
            probe_images, probe_cam, probe_view = fixed_probe
            summary, maps = probe_effect(model, probe_images.to(device), probe_cam.to(device), probe_view.to(device))
            summary.update(epoch=epoch, kind='fixed_training_image_effect_not_accuracy')
            render_probe(cfg, summary, maps, output/'effect_probes'/f'epoch_{epoch:03d}')
            with (output/'effect_probes.jsonl').open('a') as f: f.write(json.dumps(summary)+'\n')
            logger.info('CPPI fixed-sample CPPI on/off effect epoch %d: %s', epoch, summary)
            del summary, maps
            append_csv(output/'evaluation.csv', dict(epoch=epoch, **metrics))
            with (output/'evaluation_diagnostics.jsonl').open('a') as f:
                f.write(json.dumps(dict(epoch=epoch, metrics=metrics, **diagnostics))+'\n')
            logger.info('CPPI validation epoch %d: %s; %s', epoch, metrics, diagnostics)
            if metrics['mAP'] > best:
                best = metrics['mAP']; save_checkpoint(output/'transformer_best.pth', model, cfg, epoch, metrics)
        if epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS:
            save_checkpoint(output/f'transformer_{epoch}.pth', model, cfg, epoch, metrics)
