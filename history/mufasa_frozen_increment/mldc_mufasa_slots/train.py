"""Run from the TransReID root. Importing this file never starts training."""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def learning_rate(cfg, epoch):
    warmup, total = cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.MAX_EPOCHS
    if epoch <= warmup:
        fraction = (epoch-1)/max(warmup-1, 1)
        factor = cfg.SOLVER.WARMUP_FACTOR + (1-cfg.SOLVER.WARMUP_FACTOR)*fraction
    else:
        # Last epoch retains a positive LR; the schedule reaches zero afterwards.
        progress = (epoch-warmup-1)/(total-warmup)
        factor = .5*(1+math.cos(math.pi*progress))
    return cfg.SOLVER.BASE_LR*factor


def append_row(path, row):
    with Path(path).open('a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def frozen_snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in model.baseline.state_dict().items()}


def assert_frozen(model, snapshot):
    import torch
    if any(module.training for module in model.baseline.modules()):
        raise RuntimeError('A frozen MLDC submodule entered train mode')
    if any(p.requires_grad or p.grad is not None for p in model.baseline.parameters()):
        raise RuntimeError('Frozen MLDC unexpectedly has a gradient')
    for name, value in model.baseline.state_dict().items():
        if not torch.equal(snapshot[name], value.detach().cpu()):
            raise RuntimeError('Frozen MLDC parameter/buffer changed: '+name)


def save_checkpoint(cfg, model, optimizer, epoch, report, destination):
    import numpy as np
    import torch
    # The original frozen checkpoint is reused, rather than duplicated at every epoch.
    state = {name: value for name, value in model.state_dict().items() if not name.startswith('baseline.')}
    checkpoint = {'kind': 'frozen_mldc_mufasa_v1', 'epoch': epoch, 'config': cfg.dump(),
                  'state_dict': state, 'optimizer': optimizer.state_dict(),
                  'baseline_metrics': model.baseline_metrics, 'report': report,
                  'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                          'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}}
    destination = Path(destination)
    temporary = destination.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(destination)


def load_increment(model, checkpoint):
    if checkpoint.get('kind') != 'frozen_mldc_mufasa_v1':
        raise ValueError('This is not a MUFASA increment checkpoint')
    full = model.state_dict()
    expected = {name for name in full if not name.startswith('baseline.')}
    if set(checkpoint['state_dict']) != expected:
        raise RuntimeError('Increment checkpoint keys do not match the configured model')
    # Strict load with all original keys supplied also preserves BN counters;
    # loading a partial state with strict=False can reset missing BN counters.
    full.update(checkpoint['state_dict'])
    model.load_state_dict(full, strict=True)
    if checkpoint['baseline_metrics'] != model.baseline_metrics:
        raise RuntimeError('Independent MLDC reference changed')


def record_evaluation(cfg, epoch, report):
    from mldc_mufasa_slots.visualization import plot_curves
    output = Path(cfg.OUTPUT_DIR)
    for name, metrics in report['metrics'].items():
        append_row(output / 'evaluation.csv', {'epoch': epoch, 'feature': name, **metrics})
    append_row(output / 'diagnostics.csv', {'epoch': epoch, **report['diagnostics']})
    plot_curves(output)


def late_summary(output):
    with (output / 'evaluation.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    summary = {'note': 'Final checkpoint is primary; last three evaluations are stability diagnostics.'}
    for feature in ['baseline', 'final']:
        chosen = [r for r in rows if r['feature'] == feature and int(r['epoch']) > 0][-3:]
        summary[feature] = {'epochs': [int(r['epoch']) for r in chosen],
                            **{key: sum(float(r[key]) for r in chosen)/len(chosen)
                               for key in ['mAP', 'rank1', 'rank5', 'rank10']}}
    (output / 'late_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


def prepare_resume(output, epoch):
    import re
    import shutil
    checkpoints = [int(re.fullmatch(r'increment_(\d+)\.pth', p.name).group(1))
                   for p in output.glob('increment_*.pth')
                   if re.fullmatch(r'increment_(\d+)\.pth', p.name)]
    if checkpoints and max(checkpoints) > epoch:
        raise ValueError('Resume from the latest checkpoint in this output directory')
    for filename in ['training.csv', 'evaluation.csv', 'diagnostics.csv']:
        path = output / filename
        if not path.exists():
            raise FileNotFoundError('Resume needs the original run logs: '+str(path))
        with path.open(encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fields, rows = reader.fieldnames, list(reader)
        if any(int(row['epoch']) > epoch for row in rows):
            archive = output / ('resume_archive_'+time.strftime('%Y%m%d_%H%M%S'))
            archive.mkdir(exist_ok=True)
            shutil.copy2(path, archive / filename)
            with path.open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(row for row in rows if int(row['epoch']) <= epoch)


def main():
    parser = argparse.ArgumentParser(description='Frozen MLDC + MUFASA increment')
    parser.add_argument('--config_file', default='mldc_mufasa_slots/variants/slot_residual_joint_bnneck/config.yml')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume', default='', help='Resume a saved increment checkpoint')
    mode.add_argument('--evaluate', default='', help='Evaluate a saved increment without training')
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import numpy as np
    import torch
    from config import cfg as base_cfg
    from yacs.config import CfgNode as CN
    from datasets import make_dataloader
    from utils.logger import setup_logger
    from mldc_mufasa_slots.config import add_config, validate
    from mldc_mufasa_slots.model import FrozenMLDCMUFASA
    from mldc_mufasa_slots.evaluation import evaluate, assert_baseline
    from mldc_mufasa_slots.visualization import save_visualizations

    cfg = add_config(base_cfg.clone())
    checkpoint = None
    if args.resume or args.evaluate:
        checkpoint = torch.load(args.resume or args.evaluate, map_location='cpu', weights_only=False)
        cfg = CN.load_cfg(checkpoint['config'])
    else:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    validate(cfg)
    if args.resume:
        saved = CN.load_cfg(checkpoint['config'])
        current = cfg.clone()
        current.OUTPUT_DIR, current.MODEL.DEVICE_ID = saved.OUTPUT_DIR, saved.MODEL.DEVICE_ID
        if current.dump() != saved.dump():
            raise ValueError('Resume must retain the original training protocol')
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    if not torch.cuda.is_available():
        raise RuntimeError('No usable CUDA GPU. Restore the server GPU before starting training/evaluation.')
    torch.set_num_threads(cfg.MUFASA.CPU_THREADS)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True  # Same backend setting as original MLDC.
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
    logger.info('Frozen MLDC + MUFASA configuration:\n%s', cfg)
    train_loader, _, val_loader, nq, nc, cameras, views = make_dataloader(cfg)
    model = FrozenMLDCMUFASA(cfg, nc, cameras, views).cuda()
    if checkpoint is not None:
        load_increment(model, checkpoint)
    if args.evaluate:
        epoch = checkpoint['epoch']
        destination = output / 'checkpoint_analysis' / f'epoch_{epoch:03d}'
        report = evaluate(cfg, model, val_loader, nq, destination)
        assert_baseline(report['metrics']['baseline'], model.baseline_metrics)
        logger.info('Checkpoint %d evaluation: %s', epoch, json.dumps(report['metrics']))
        if cfg.MUFASA.VISUALIZE:
            save_visualizations(cfg, model, val_loader, nq, epoch)
        return
    from mldc_mufasa_slots.model import compute_loss
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    start_epoch = 1
    snapshot = frozen_snapshot(model)
    metadata = {'config': cfg.dump(), 'baseline_metrics': model.baseline_metrics,
                'trainable_parameters': sum(p.numel() for p in parameters),
                'frozen_parameters': sum(p.numel() for p in model.baseline.parameters()),
                'precision': 'FP32, including the original frozen MLDC',
                'primary_checkpoint': 'increment_120.pth',
                'implementation_notes': ['SA bi-level truncation, deterministic embedding initialization',
                    'SA input LayerNorm; omit inactive outer input MLP in the official default code',
                    'Exact max-IoU assignment over 24 permutations for four slots',
                    'Adjacent-pair means and M-Fusion; no learned mask fusion',
                    'Training-only four-hidden-layer spatial-broadcast MLP decoder',
                    'ID and triplet act only on the final normalized 768D descriptor']}
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']+1
        logger.info('Resuming after epoch %d', checkpoint['epoch'])
    else:
        (output / 'config.yml').write_text(cfg.dump(), encoding='utf-8')
        (output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        logger.info('Checking original full-query/gallery baseline before any optimizer step')
    # On resume this also checks the frozen baseline, without duplicating curve rows.
    check_epoch = start_epoch-1
    report = evaluate(cfg, model, val_loader, nq, output / 'evaluations' / f'epoch_{check_epoch:03d}')
    assert_baseline(report['metrics']['baseline'], model.baseline_metrics)
    assert_frozen(model, snapshot)
    logger.info('Preflight baseline reproduced: %s', json.dumps(report['metrics']['baseline']))
    if checkpoint is None:
        record_evaluation(cfg, 0, report)
        if cfg.MUFASA.VISUALIZE:
            save_visualizations(cfg, model, val_loader, nq, 0)
    else:
        state = checkpoint['rng']
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        torch.cuda.set_rng_state_all(state['cuda'])
        del checkpoint
    logger.info('Training only %d new parameters; frozen original MLDC is always in eval mode',
                metadata['trainable_parameters'])
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS+1):
        started = time.monotonic()
        model.train()
        lr = learning_rate(cfg, epoch)
        for group in optimizer.param_groups:
            group['lr'] = lr
        sums = {'loss': 0., 'identity': 0., 'triplet': 0., 'reconstruction': 0., 'accuracy': 0.}
        count = 0
        for iteration, (images, labels, camera, view) in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            labels = labels.cuda()
            parts = model(images.cuda(), cam_label=camera.cuda(), view_label=view.cuda())
            loss, terms = compute_loss(cfg, parts, labels)
            loss.backward()
            optimizer.step()
            size = len(labels)
            sums['loss'] += loss.item()*size
            for key, value in terms.items():
                sums[key] += value.item()*size
            sums['accuracy'] += (parts['logits'].argmax(-1) == labels).sum().item()
            count += size
            if iteration % cfg.SOLVER.LOG_PERIOD == 0:
                logger.info('Epoch[%d] Iter[%d/%d] Loss %.5f, ID %.5f, Tri %.5f, Rec %.5f, Acc %.4f, LR %.3e',
                            epoch, iteration, len(train_loader), sums['loss']/count, sums['identity']/count,
                            sums['triplet']/count, sums['reconstruction']/count, sums['accuracy']/count, lr)
            del parts, loss, terms
        append_row(output / 'training.csv', {'epoch': epoch, 'lr': lr,
                   **{key: value/count for key, value in sums.items()}, 'seconds': time.monotonic()-started})
        report = None
        if epoch % cfg.SOLVER.EVAL_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS:
            report = evaluate(cfg, model, val_loader, nq, output / 'evaluations' / f'epoch_{epoch:03d}')
            assert_baseline(report['metrics']['baseline'], model.baseline_metrics)
            record_evaluation(cfg, epoch, report)
            logger.info('Epoch %d retrieval: %s', epoch, json.dumps(report['metrics']))
            logger.info('Fixed queries: %d; broken queries: %d', report['diagnostics']['fixed_queries'],
                        report['diagnostics']['broken_queries'])
        if cfg.MUFASA.VISUALIZE and (epoch % cfg.MUFASA.VIS_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS):
            save_visualizations(cfg, model, val_loader, nq, epoch)
        if epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS:
            assert_frozen(model, snapshot)
            save_checkpoint(cfg, model, optimizer, epoch, report, output / f'increment_{epoch}.pth')
            logger.info('Saved epoch %d; original MLDC parameters and buffers unchanged', epoch)
    late_summary(output)
    logger.info('Training complete. Primary result: increment_%d.pth', cfg.SOLVER.MAX_EPOCHS)


if __name__ == '__main__':
    main()
