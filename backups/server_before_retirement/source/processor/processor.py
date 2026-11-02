import logging
import os
import random
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist


def _layer_scale_log(model):
    """Report the trained per-channel LayerScale parameters, not their initial setting."""
    model = model.module if hasattr(model, 'module') else model
    readout = getattr(getattr(model, 'base', None), 'cls_readout', None)
    if readout is None:
        return ''
    with torch.no_grad():
        ca = readout.gamma_attn.detach().float()
        ffn = readout.gamma_mlp.detach().float()
        return (', LayerScale CA mean/max|.|: {:.3e}/{:.3e}'
                ', FFN mean/max|.|: {:.3e}/{:.3e}').format(
                    ca.mean().item(), ca.abs().max().item(),
                    ffn.mean().item(), ffn.abs().max().item())


def _save_fusion_heatmaps(cfg, model, val_loader, epoch, logger):
    """Draw three randomly chosen validation images without retaining training graphs."""
    model = model.module if hasattr(model, 'module') else model
    backbone = getattr(model, 'base', None)
    if backbone is None or not getattr(backbone, 'multilayer_fusion', False):
        logger.info('Fusion heatmaps skipped: multi-layer fusion is disabled.')
        return

    # The validation dataset uses deterministic resize/normalization. Direct indexing
    # samples the whole query+gallery set without forwarding another full val epoch.
    dataset = val_loader.dataset
    indices = random.SystemRandom().sample(range(len(dataset)), min(3, len(dataset)))
    if not indices:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm as cm
    import numpy as np

    output_dir = os.path.join(cfg.OUTPUT_DIR, 'fusion_heatmaps',
                              'epoch_{:03d}'.format(epoch))
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    for sample_number, dataset_index in enumerate(indices, start=1):
        img, pid, camid, viewid, img_path = dataset[dataset_index]
        captured = {}

        def capture_original_attention(module, inputs):
            tokens = inputs[0]
            batch, length, dim = tokens.shape
            qkv = module.qkv(tokens).reshape(
                batch, length, 3, module.num_heads, dim // module.num_heads
            ).permute(2, 0, 3, 1, 4)
            probabilities = ((qkv[0] @ qkv[1].transpose(-2, -1))
                             * module.scale).float().softmax(dim=-1)
            captured['before_attention'] = probabilities[:, :, 0, 1:].mean(1)[0].cpu()

        def capture_fused_attention(module, inputs):
            tokens = inputs[0]
            batch, length, dim = tokens.shape
            q = module.q(tokens[:, :1]).reshape(
                batch, 1, module.num_heads, dim // module.num_heads).transpose(1, 2)
            k = module.k(tokens).reshape(
                batch, length, module.num_heads, dim // module.num_heads).transpose(1, 2)
            probabilities = ((q * module.scale) @ k.transpose(-2, -1)).float().softmax(dim=-1)
            captured['after_attention'] = probabilities[:, :, 0, 1:].mean(1)[0].cpu()

        def capture_fusion(module, inputs, output):
            patch_tokens, grid_size = inputs
            captured['grid_size'] = grid_size
            captured['layer_norms'] = [tokens[0].float().norm(dim=-1).cpu()
                                       for tokens in patch_tokens]
            captured['fused_norm'] = output[0].float().norm(dim=-1).cpu()

        hooks = [
            backbone.blocks[-1].attn.register_forward_pre_hook(capture_original_attention),
            backbone.cls_readout.attn.register_forward_pre_hook(capture_fused_attention),
            backbone.patch_fusion.register_forward_hook(capture_fusion),
        ]
        try:
            with torch.no_grad():
                model(img.unsqueeze(0).to(device),
                      cam_label=torch.tensor([camid], device=device),
                      view_label=torch.tensor([viewid], device=device))
        finally:
            for hook in hooks:
                hook.remove()

        height, width = captured['grid_size']
        before = captured['before_attention']
        after = captured['after_attention']
        # Exclude the CLS self-key from both distributions for a spatial comparison.
        before = (before / before.sum().clamp_min(1e-12)).reshape(height, width).numpy()
        after = (after / after.sum().clamp_min(1e-12)).reshape(height, width).numpy()
        difference = after - before
        norms = [values.reshape(height, width).numpy()
                 for values in captured['layer_norms']]
        norms.append(captured['fused_norm'].reshape(height, width).numpy())
        attention_max = max(before.max(), after.max())
        difference_max = max(abs(difference.min()), abs(difference.max()), 1e-12)
        mean = torch.tensor(cfg.INPUT.PIXEL_MEAN).view(3, 1, 1)
        std = torch.tensor(cfg.INPUT.PIXEL_STD).view(3, 1, 1)
        image = (img.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        image_height, image_width = image.shape[:2]

        def show_overlay(ax, values, cmap, low, high, signed=False):
            # The ViT grid is coarse; bicubic resizing smooths its DISPLAY only.
            # It cannot recover spatial detail that was not in the patch tokens.
            smooth = torch.nn.functional.interpolate(
                torch.from_numpy(values.copy()).float()[None, None],
                size=(image_height, image_width), mode='bicubic',
                align_corners=False)[0, 0].numpy()
            scale = max(high - low, 1e-12)
            color_position = np.clip((smooth - low) / scale, 0, 1)
            strength = (np.clip(np.abs(smooth) / max(abs(low), abs(high), 1e-12), 0, 1)
                        if signed else color_position)
            rgba = matplotlib.colormaps[cmap](color_position)
            rgba[..., 3] = 0.70 * strength
            ax.imshow(image)
            ax.imshow(rgba, interpolation='bilinear')

        fig, axes = plt.subplots(2, 4, figsize=(15, 9), constrained_layout=True)
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original night image')
        norm_axes = [axes[0, 1], axes[0, 2], axes[0, 3], axes[1, 0]]
        for ax, values, label in zip(norm_axes, norms,
                                     ['Block {:02d} patch norm'.format(layer)
                                      for layer in backbone.fusion_layers] + ['Fused patch norm']):
            # Norm contrast is scaled per image; numeric raw range stays visible.
            show_overlay(ax, values, 'turbo', float(values.min()), float(values.max()))
            ax.set_title('{}\nL2 {:.2f}–{:.2f}'.format(
                label, float(values.min()), float(values.max())))
        for ax, values, label in [(axes[1, 1], before, 'Before: last-block CLS attention'),
                                  (axes[1, 2], after, 'After: fused CLS attention')]:
            # The attention pair keeps one common scale; a flat after-map stays flat.
            show_overlay(ax, values, 'turbo', 0, attention_max)
            ax.set_title(label)
        fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(0, attention_max), cmap='turbo'),
                     ax=[axes[1, 1], axes[1, 2]], shrink=0.55,
                     label='Patch attention (shared scale)')
        show_overlay(axes[1, 3], difference, 'coolwarm',
                     -difference_max, difference_max, signed=True)
        axes[1, 3].set_title('After - before attention')
        fig.colorbar(cm.ScalarMappable(
            norm=colors.Normalize(-difference_max, difference_max), cmap='coolwarm'),
                     ax=axes[1, 3], shrink=0.55,
                     label='Attention difference')
        for ax in axes.flat:
            ax.axis('off')
        fig.suptitle('Epoch {} | validation index {} | PID {} | {}\n'
                     'Heatmaps over original image; bicubic display, no new spatial detail. '
                     'Patch-norm contrast is scaled per panel.'.format(
                         epoch, dataset_index, pid, os.path.basename(str(img_path))))
        filename = os.path.join(output_dir, 'sample_{:02d}_index_{:05d}.png'.format(
            sample_number, dataset_index))
        fig.savefig(filename, dpi=120)
        plt.close(fig)
        logger.info('Saved fusion heatmap: %s', filename)


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view )
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    if (n_iter + 1) % log_period == 0:
                        base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                        logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}{}"
                                    .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg,
                                            acc_meter.avg, base_lr, _layer_scale_log(model)))
            else:
                if (n_iter + 1) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}{}"
                                .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg,
                                        acc_meter.avg, base_lr, _layer_scale_log(model)))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter + 1), train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save((model.module if hasattr(model, 'module') else model).state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save((model.module if hasattr(model, 'module') else model).state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                dist.barrier()
                if dist.get_rank() == 0:
                    model.eval()
                    eval_model = model.module if hasattr(model, 'module') else model
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = eval_model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
                dist.barrier()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

        if epoch % 10 == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    _save_fusion_heatmaps(cfg, model, val_loader, epoch, logger)
                dist.barrier()
            else:
                _save_fusion_heatmaps(cfg, model, val_loader, epoch, logger)


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]

