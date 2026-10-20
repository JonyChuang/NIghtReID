import logging
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist


def _similarity_heatmap(feature, reference, patch_tokens, patch_grid):
    """Spatial magnitude of grad x activation for a same-ID similarity score."""
    score = F.cosine_similarity(feature.float(), reference.detach().float(), dim=1).sum()
    gradient = torch.autograd.grad(score, patch_tokens, retain_graph=True)[0]
    contribution = (patch_tokens.float() * gradient.float()).abs().mean(dim=-1)
    height, width = patch_grid
    if contribution.shape[1] != height * width:
        raise ValueError('Patch grid does not match the backbone patch tokens')
    return contribution.reshape(1, 1, height, width).detach(), score.detach().item()


def _save_ffvt_heatmaps(cfg, model, val_loader, num_query, epoch, logger):
    """Overlay global/fusion attribution in the selected retrieval space."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    current = model.module if hasattr(model, 'module') else model
    records = val_loader.dataset.dataset
    gallery_by_pid = {}
    for index in range(num_query, len(records)):
        gallery_by_pid.setdefault(records[index][1], []).append(index)
    candidates = [index for index in range(num_query)
                  if any(records[g][2] != records[index][2]
                         for g in gallery_by_pid.get(records[index][1], ()))]
    if not candidates:
        logger.warning('FFVT heatmaps: no cross-camera same-ID query/gallery pairs')
        return

    rng = random.Random(int(cfg.SOLVER.SEED) + epoch)
    chosen = rng.sample(candidates, min(3, len(candidates)))
    destination = os.path.join(cfg.OUTPUT_DIR, 'ffvt_heatmaps', 'epoch_{:03d}'.format(epoch))
    os.makedirs(destination, exist_ok=True)
    device = next(current.parameters()).device
    patch_embed = current.base.patch_embed
    grid = (patch_embed.num_y, patch_embed.num_x)
    was_training = current.training
    current.eval()
    try:
        for number, query_index in enumerate(chosen, 1):
            query = records[query_index]
            matches = [g for g in gallery_by_pid[query[1]] if records[g][2] != query[2]]
            reference_index = rng.choice(matches)
            reference = records[reference_index]
            query_image, _, _, query_view, _ = val_loader.dataset[query_index]
            reference_image, _, _, reference_view, _ = val_loader.dataset[reference_index]
            query_image = query_image.unsqueeze(0).to(device)
            reference_image = reference_image.unsqueeze(0).to(device)
            query_cam = torch.tensor([query[2]], device=device)
            reference_cam = torch.tensor([reference[2]], device=device)
            query_view = torch.tensor([query_view], device=device)
            reference_view = torch.tensor([reference_view], device=device)

            with torch.no_grad():
                _, reference_cls, reference_alf, reference_parts = current(
                    reference_image, cam_label=reference_cam,
                    view_label=reference_view, return_parts=True)
                if current.neck_feat == 'after':
                    reference_cls = current.bottleneck_cls(reference_cls)
                    reference_parts = torch.stack([
                        neck(reference_parts[:, index])
                        for index, neck in enumerate(current.bottleneck_parts)], dim=1)
                    reference_alf = reference_parts.flatten(1)

            captured = []
            def capture_patch_tokens(module, inputs, output):
                if not output.requires_grad:
                    output = output.detach().requires_grad_(True)
                captured.append(output)
                return output

            hook = patch_embed.register_forward_hook(capture_patch_tokens)
            try:
                with torch.enable_grad():
                    _, final_cls, alf_feature, part_features = current(
                        query_image, cam_label=query_cam,
                        view_label=query_view, return_parts=True)
                    if current.neck_feat == 'after':
                        final_cls = current.bottleneck_cls(final_cls)
                        part_features = torch.stack([
                            neck(part_features[:, index])
                            for index, neck in enumerate(current.bottleneck_parts)], dim=1)
                        alf_feature = part_features.flatten(1)
                    patch_tokens = captured[0]
                    alf_map, alf_score = _similarity_heatmap(
                        alf_feature, reference_alf, patch_tokens, grid)
                    cls_map, cls_score = _similarity_heatmap(
                        final_cls, reference_cls, patch_tokens, grid)
                    part_maps = [
                        _similarity_heatmap(part_features[:, index],
                                            reference_parts[:, index], patch_tokens, grid)
                        for index in range(current.num_parts)] if current.num_parts > 1 else []
            finally:
                hook.remove()

            with Image.open(query[0]) as source:
                original = source.convert('RGB')
            with Image.open(reference[0]) as source:
                positive = source.convert('RGB')
            width, height = original.size

            def smooth_map(values):
                values = F.interpolate(values.float(), size=(height, width),
                                       mode='bicubic', align_corners=False)
                values = values.clamp_min(0).squeeze().cpu().numpy()
                upper = float(np.percentile(values, 99))
                return np.clip(values / upper, 0, 1) if upper > 0 else np.zeros_like(values)

            columns = 4
            rows = (4 + len(part_maps) + columns - 1) // columns
            fig, axes = plt.subplots(rows, columns, figsize=(12, 5 * rows),
                                     constrained_layout=True, squeeze=False)
            axes = axes.ravel()
            axes[0].imshow(original)
            axes[0].set_title('Night query | PID {} / cam {}'.format(query[1], query[2]))
            axes[1].imshow(positive)
            axes[1].set_title('Same ID | cam {}'.format(reference[2]))
            panels = [(axes[2], cls_map, 'Block 12 CLS output', cls_score),
                      (axes[3], alf_map, 'FFVT fusion CLS' if current.num_parts == 1
                       else 'FFVT fusion features', alf_score)]
            panels.extend((axes[4 + index], values, 'Part query {}'.format(index + 1), score)
                          for index, (values, score) in enumerate(part_maps))
            for axis, values, title, score in panels:
                intensity = smooth_map(values)
                colors = plt.get_cmap('turbo')(intensity)
                colors[..., 3] = 0.65 * np.power(intensity, 0.7)
                axis.imshow(original)
                axis.imshow(colors, interpolation='bilinear', extent=(0, width, height, 0))
                axis.set_title('{}\ncosine {:.3f}'.format(title, score))
            for axis in axes[4 + len(part_maps):]:
                axis.set_visible(False)
            for axis in axes:
                axis.axis('off')
            fig.suptitle('Epoch {} | Features {} BN | Same-ID cross-camera cosine: |gradient x activation| at patch embeddings\n'
                         'Fixed MAWS selection; bicubic display; colors scaled independently per map'.format(epoch, current.neck_feat),
                         fontsize=10)
            filename = 'sample_{:02d}_query_{:05d}_pid_{}.png'.format(
                number, query_index, query[1])
            fig.savefig(os.path.join(destination, filename), dpi=160)
            plt.close(fig)
            # Store the query's actual MAWS scores and zero-based patch indices.
            # The heatmaps are gradients with the discrete selection held fixed.
            selection_data = {
                'query_index': np.asarray(query_index),
                'reference_index': np.asarray(reference_index),
                'patch_grid': np.asarray(grid),
                'layers': np.asarray(current.base.fusion_layers),
                'neck_feat': np.asarray(current.neck_feat),
                'cls_attribution': cls_map.cpu().numpy(),
                'fusion_attribution': alf_map.cpu().numpy(),
            }
            for layer, selection in current.base.last_selection.items():
                selection_data['layer_{}_patch_indices'.format(layer)] = (
                    selection['indices'][0].cpu().numpy() - 1)
                selection_data['layer_{}_maws'.format(layer)] = (
                    selection['scores'][0, 1:].cpu().numpy())
            np.savez_compressed(os.path.join(destination, filename.replace('.png', '.npz')),
                                **selection_data)
        logger.info('Saved {} global/fusion overlays and MAWS selections to {}'.format(
            len(chosen), destination))
    finally:
        current.train(was_training)


def _ffvt_component_log(model):
    current = model.module if hasattr(model, 'module') else model
    ratio = current.last_fusion_cls_ratio
    if ratio is None:
        return ''
    return ', Fusion/CLS norm: {:.4f}'.format(ratio.detach().float().item())


def _ffvt_learning_rates(model, optimizer):
    current = model.module if hasattr(model, 'module') else model
    parameters = (current.base.patch_embed.proj.weight,
                  current.fusion.block.attn.qkv.weight,
                  current.classifier_cls.weight)
    rates = {}
    for group in optimizer.param_groups:
        for parameter in group['params']:
            rates[id(parameter)] = group['lr']
    return tuple(rates[id(parameter)] for parameter in parameters)


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
    logger.info('start FFVT dual-branch training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    part_acc_meter = AverageMeter()
    term_meters = {name: AverageMeter()
                   for name in ('cls_id', 'parts_id', 'cls_tri', 'parts_tri')}

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        part_acc_meter.reset()
        for meter in term_meters.values():
            meter.reset()
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
            # Acc and Fusion Acc use the two independent identity classifiers.
            acc = (score['cls'].detach().argmax(dim=1) == target).float().mean()
            part_acc = torch.stack([
                (logits.detach().argmax(dim=1) == target).float().mean()
                for logits in score['parts']]).mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc.item(), img.shape[0])
            part_acc_meter.update(part_acc.item(), img.shape[0])
            for name, value in loss_fn.last_terms.items():
                term_meters[name].update(value.item(), img.shape[0])

            torch.cuda.synchronize()
            if ((not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0)
                    and (n_iter + 1) % log_period == 0):
                base_lr, fusion_lr, head_lr = _ffvt_learning_rates(model, optimizer)
                logger.info(
                    'Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Fusion Acc: {:.3f}, '
                    'CLS ID: {:.3f}, Fusion ID: {:.3f}, CLS Tri: {:.3f}, Fusion Tri: {:.3f}, '
                    'Base Lr: {:.2e}, Fusion Lr: {:.2e}, Head Lr: {:.2e}{}'.format(
                        epoch, n_iter + 1, len(train_loader), loss_meter.avg,
                        acc_meter.avg, part_acc_meter.avg,
                        term_meters['cls_id'].avg, term_meters['parts_id'].avg,
                        term_meters['cls_tri'].avg, term_meters['parts_tri'].avg,
                        base_lr, fusion_lr, head_lr, _ffvt_component_log(model)))

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
                if dist.get_rank() == 0:
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
                    if epoch % 10 == 0:
                        try:
                            _save_ffvt_heatmaps(cfg, model, val_loader, num_query, epoch, logger)
                        except Exception:
                            logger.exception('FFVT heatmap generation failed at epoch %s', epoch)
                    torch.cuda.empty_cache()
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
                if epoch % 10 == 0:
                    try:
                        _save_ffvt_heatmaps(cfg, model, val_loader, num_query, epoch, logger)
                    except Exception:
                        logger.exception('FFVT heatmap generation failed at epoch %s', epoch)
                torch.cuda.empty_cache()


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

