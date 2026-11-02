"""Joint MLDC+MUFASA entry. Importing this module never starts training."""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
CHECKPOINT_KIND = 'joint_mldc_mufasa_v2'


def append_row(path, row):
    with Path(path).open('a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def make_training(cfg, model, num_classes):
    from solver import make_optimizer
    from solver.scheduler_factory import create_scheduler
    from mldc_cls_readout.loss.make_loss_mldc import make_loss
    _, center = make_loss(cfg, num_classes)
    optimizer, _ = make_optimizer(cfg, model, center)
    prefixes = ('mldc_model.bottleneck.', 'mldc_model.classifier.', 'classifier.')
    heads = {id(p) for name, p in model.named_parameters()
             if p.requires_grad and name.startswith(prefixes)}
    for group in optimizer.param_groups:
        ids = {id(p) for p in group['params']}
        if ids & heads:
            if not ids <= heads:
                raise RuntimeError('Optimizer mixes head and non-head parameters')
            group['lr'] *= cfg.SOLVER.MLDC_HEAD_LR_FACTOR
    return optimizer, create_scheduler(cfg, optimizer)


def load_joint(model, checkpoint):
    if checkpoint.get('kind') != CHECKPOINT_KIND:
        raise ValueError('Expected a joint v2 checkpoint; frozen increments cannot resume this model')
    model.load_state_dict(checkpoint['state_dict'], strict=True)


def save_checkpoint(cfg, model, optimizer, scheduler, scaler, epoch, report, best_map, destination):
    import numpy as np
    import torch
    checkpoint = {'kind': CHECKPOINT_KIND, 'epoch': epoch, 'config': cfg.dump(),
                  'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                  'report': report, 'best_map': best_map,
                  'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                          'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}}
    destination = Path(destination)
    temporary = destination.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(destination)


def record_evaluation(cfg, epoch, report):
    from mldc_mufasa_slots.visualization import plot_curves
    output = Path(cfg.OUTPUT_DIR)
    for name, metrics in report['metrics'].items():
        append_row(output / 'evaluation.csv', {'epoch': epoch, 'feature': name, **metrics})
    append_row(output / 'diagnostics.csv', {'epoch': epoch, **report['diagnostics']})
    plot_curves(output)


def prepare_resume(output, epoch):
    import re
    import shutil
    numbered = [int(m.group(1)) for p in output.glob('transformer_*.pth')
                if (m := re.fullmatch(r'transformer_(\d+)\.pth', p.name))]
    if numbered and max(numbered) > epoch:
        raise ValueError('Resume from the latest checkpoint in this output directory')
    archive = output / ('resume_archive_' + time.strftime('%Y%m%d_%H%M%S'))
    for filename in ['training.csv', 'evaluation.csv', 'diagnostics.csv', 'layer_weights.csv']:
        path = output / filename
        if not path.exists():
            raise FileNotFoundError('Resume needs the original run logs: ' + str(path))
        with path.open(encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fields, rows = reader.fieldnames, list(reader)
        if any(int(r['epoch']) > epoch for r in rows):
            archive.mkdir(exist_ok=True)
            shutil.copy2(path, archive / filename)
            with path.open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(r for r in rows if int(r['epoch']) <= epoch)


def late_summary(output):
    with (output / 'evaluation.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    summary = {'note': 'mldc is the current joint branch, not the independent frozen baseline.'}
    for name in ['mldc', 'final']:
        chosen = [r for r in rows if r['feature'] == name][-3:]
        if chosen:
            summary[name] = {'epochs': [int(r['epoch']) for r in chosen],
                             **{k: sum(float(r[k]) for r in chosen)/len(chosen)
                                for k in ['mAP', 'rank1', 'rank5', 'rank10']}}
    (output / 'late_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Joint MLDC + MUFASA from SSL pretraining')
    parser.add_argument('--config_file', default='mldc_mufasa_slots/variants/slot_residual_joint_bnneck/config.yml')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume', default='')
    mode.add_argument('--evaluate', default='')
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import numpy as np
    import torch
    from torch.cuda import amp
    from config import cfg as base_cfg
    from yacs.config import CfgNode as CN
    from datasets import make_dataloader
    from utils.logger import setup_logger
    from mldc_cls_readout.processor.processor_mldc import record_weights
    from mldc_mufasa_slots.config import add_config, validate
    from mldc_mufasa_slots.model import JointMLDCMUFASA, compute_loss
    from mldc_mufasa_slots.evaluation import evaluate
    from mldc_mufasa_slots.visualization import save_visualizations

    checkpoint = None
    if args.resume or args.evaluate:
        checkpoint = torch.load(args.resume or args.evaluate, map_location='cpu', weights_only=False)
        if checkpoint.get('kind') != CHECKPOINT_KIND:
            raise ValueError('Frozen increment/old architecture cannot load as a joint v2 experiment')
        cfg = CN.load_cfg(checkpoint['config'])
    else:
        cfg = add_config(base_cfg.clone())
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    validate(cfg)
    if args.resume:
        saved = CN.load_cfg(checkpoint['config'])
        current = cfg.clone()
        current.OUTPUT_DIR, current.MODEL.DEVICE_ID = saved.OUTPUT_DIR, saved.MODEL.DEVICE_ID
        if current.dump() != saved.dump():
            raise ValueError('Resume must retain the joint training protocol')
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    if not torch.cuda.is_available():
        raise RuntimeError('No usable CUDA GPU; no training started')
    torch.set_num_threads(cfg.MUFASA.CPU_THREADS)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    seed = cfg.SOLVER.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    output = Path(cfg.OUTPUT_DIR)
    if not args.evaluate and not args.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError('Output is not empty; use --resume or a different OUTPUT_DIR')
    if args.resume:
        prepare_resume(output, checkpoint['epoch'])
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('transreid', str(output), if_train=not bool(args.evaluate))
    logger.info('Joint MLDC + MUFASA configuration:\n%s', cfg)
    train_loader, _, val_loader, nq, nc, cameras, views = make_dataloader(cfg)
    model = JointMLDCMUFASA(cfg, nc, cameras, views).cuda()
    if checkpoint is not None:
        load_joint(model, checkpoint)
    if args.evaluate:
        epoch = checkpoint['epoch']
        report = evaluate(cfg, model, val_loader, nq, output / 'checkpoint_analysis' / f'epoch_{epoch:03d}')
        logger.info('Checkpoint %d evaluation: %s', epoch, json.dumps(report['metrics']))
        if cfg.MUFASA.VISUALIZE:
            save_visualizations(cfg, model, val_loader, nq, epoch)
        return
    optimizer, scheduler = make_training(cfg, model, nc)
    scaler = amp.GradScaler()
    start_epoch, best_map = 1, -1.0
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch, best_map = checkpoint['epoch']+1, checkpoint['best_map']
        state = checkpoint['rng']
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        torch.cuda.set_rng_state_all(state['cuda'])
        del checkpoint
        logger.info('Resuming after epoch %d', start_epoch-1)
    else:
        (output / 'config.yml').write_text(cfg.dump(), encoding='utf-8')
        metadata = {'kind': CHECKPOINT_KIND, 'config': cfg.dump(), 'initialization': cfg.MODEL.PRETRAIN_PATH,
                    'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                    'precision': 'Baseline AMP ViT; FP32 slot modules and losses',
                    'loss': 'ReID(global) + weighted ReID(final) + residual metric + reconstruction',
                    'loss_weights': {'global_ID': cfg.MODEL.ID_LOSS_WEIGHT,
                                     'global_triplet': cfg.MODEL.TRIPLET_LOSS_WEIGHT,
                                     'final_ReID': cfg.MUFASA.FINAL_LOSS_WEIGHT,
                                     'residual_metric': cfg.MUFASA.RESIDUAL_METRIC_WEIGHT,
                                     'reconstruction': cfg.MUFASA.RECON_WEIGHT},
                    'diagnostics': 'mldc is the jointly trained branch, not the historical independent baseline'}
        (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        record_weights(model, 0, output, logger)
        if cfg.MUFASA.VISUALIZE:
            save_visualizations(cfg, model, val_loader, nq, 0)
    logger.info('All active ViT/MLDC/MUFASA modules train together from epoch 1')
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS+1):
        started = time.monotonic()
        model.train()
        lr = optimizer.param_groups[0]['lr']
        sums = {k: 0. for k in ['loss', 'global_identity', 'global_triplet', 'identity', 'triplet',
                                'reconstruction', 'residual_metric', 'accuracy', 'global_accuracy', 'fusion_scale']}
        count = 0
        for iteration, (images, labels, camera, view) in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            labels = labels.cuda()
            with amp.autocast():
                parts = model(images.cuda(), cam_label=camera.cuda(), view_label=view.cuda())
                loss, terms = compute_loss(cfg, parts, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            size = len(labels)
            sums['loss'] += loss.item()*size
            for key, value in terms.items():
                sums[key] += value.item()*size
            sums['accuracy'] += (parts['logits'].argmax(-1) == labels).sum().item()
            sums['global_accuracy'] += (parts['global_logits'].argmax(-1) == labels).sum().item()
            sums['fusion_scale'] += parts['fusion_scale'].detach().sum().item()
            count += size
            if iteration % cfg.SOLVER.LOG_PERIOD == 0:
                logger.info('Epoch[%d] Iter[%d/%d] Loss %.5f, Global ID/Tri %.5f/%.5f, '
                            'Final ID/Tri %.5f/%.5f, Residual Tri %.5f, Rec %.5f, '
                            'Acc %.4f/%.4f, Gate %.4f, Base LR %.3e',
                            epoch, iteration, len(train_loader), sums['loss']/count,
                            sums['global_identity']/count, sums['global_triplet']/count,
                            sums['identity']/count, sums['triplet']/count, sums['residual_metric']/count,
                            sums['reconstruction']/count, sums['global_accuracy']/count,
                            sums['accuracy']/count, sums['fusion_scale']/count, lr)
            del parts, loss, terms
        # Same scheduler and epoch-end timing as independent MLDC.
        scheduler.step(epoch)
        append_row(output / 'training.csv', {'epoch': epoch, 'lr': lr,
                   **{k: v/count for k, v in sums.items()}, 'seconds': time.monotonic()-started})
        record_weights(model, epoch, output, logger)
        report, improved = None, False
        if epoch % cfg.SOLVER.EVAL_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS:
            report = evaluate(cfg, model, val_loader, nq, output / 'evaluations' / f'epoch_{epoch:03d}')
            record_evaluation(cfg, epoch, report)
            logger.info('Epoch %d retrieval: %s', epoch, json.dumps(report['metrics']))
            logger.info('Versus current MLDC: fixed %d; broken %d', report['diagnostics']['fixed_queries'],
                        report['diagnostics']['broken_queries'])
            improved = report['metrics']['final']['mAP'] > best_map
            if improved:
                best_map = report['metrics']['final']['mAP']
                (output / 'best_metrics.json').write_text(json.dumps({'epoch': epoch,
                    **report['metrics']['final']}, indent=2), encoding='utf-8')
        if cfg.MUFASA.VISUALIZE and (epoch % cfg.MUFASA.VIS_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS):
            save_visualizations(cfg, model, val_loader, nq, epoch)
        if improved:
            save_checkpoint(cfg, model, optimizer, scheduler, scaler, epoch, report, best_map,
                            output / 'transformer_best.pth')
        if report is not None:
            save_checkpoint(cfg, model, optimizer, scheduler, scaler, epoch, report, best_map,
                            output / 'transformer_latest.pth')
        if epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS:
            save_checkpoint(cfg, model, optimizer, scheduler, scaler, epoch, report, best_map,
                            output / f'transformer_{epoch}.pth')
    late_summary(output)
    logger.info('Joint training complete. Final checkpoint: transformer_%d.pth', cfg.SOLVER.MAX_EPOCHS)


if __name__ == '__main__':
    main()
