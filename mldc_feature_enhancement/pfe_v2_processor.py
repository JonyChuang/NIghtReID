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
from .pfe_processor import evaluate
from .pfe_v2_optim import lr_summary
from .pfe_v2_diagnostics import probe_effect, render_probe


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
                logger.info('V2 startup actual effect: %s', summary)
                del summary, maps
            inspect = iteration == 1
            with torch.autocast('cuda'):
                data = model(images, labels, cam_label=camera.to(device), view_label=view.to(device), diagnostics=inspect)
                loss, identity, triplet = objectives(data, labels, cfg)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            if inspect:
                snapshots = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith('pfe_adapters.')}
                detail = dict(epoch=epoch, lr=lr_summary(optimizer),
                    stage_ratio_mean=data['pfe_correction_ratio'].mean(0).cpu().tolist())
                detail['gradient_norms'] = {part: sum(float(p.grad.float().square().sum()) for n,p in model.named_parameters()
                    if n.startswith('pfe_adapters.') and part in n and p.grad is not None)**.5
                    for part in ('.down.', '.block.', '.up.')}
            if iteration == 1 and epoch == 1:
                norms = {prefix: sum(float(p.grad.float().square().sum()) for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None) ** .5
                         for prefix in ('pfe_adapters.', 'base.patch_embed.', 'base.blocks.', 'classifier.')}
                logger.info('PFE joint gradient norms: %s', norms)
            before = model.classifier.weight._version
            scaler.step(optimizer); scaler.update()
            updates += int(model.classifier.weight._version != before); batches += 1
            if inspect:
                detail['optimizer_step_applied'] = model.classifier.weight._version != before
                detail['relative_updates'] = {}
                for part in ('.down.', '.block.', '.up.'):
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
                with (output/'pfe_learning_diagnostics.jsonl').open('a') as f: f.write(json.dumps(detail)+'\n')
                logger.info('V2 magnitude / LR / actual parameter update: %s', detail)
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
            logger.info('V2 fixed-sample PFE on/off effect epoch %d: %s', epoch, summary)
            del summary, maps
            append_csv(output/'evaluation.csv', dict(epoch=epoch, **metrics))
            with (output/'evaluation_diagnostics.jsonl').open('a') as f:
                f.write(json.dumps(dict(epoch=epoch, metrics=metrics, **diagnostics))+'\n')
            logger.info('PFE validation epoch %d: %s; %s', epoch, metrics, diagnostics)
            if metrics['mAP'] > best:
                best = metrics['mAP']; save_checkpoint(output/'transformer_best.pth', model, cfg, epoch, metrics)
        if epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS:
            save_checkpoint(output/f'transformer_{epoch}.pth', model, cfg, epoch, metrics)
