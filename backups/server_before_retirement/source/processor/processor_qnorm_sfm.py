import copy
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda import amp

from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval


def _model_core(model):
    return model.module if hasattr(model, "module") else model


def _load_teacher(cfg, student, device):
    """Load the verified Night600 checkpoint into an independent frozen copy."""
    teacher_path = cfg.MODEL.TEACHER_WEIGHT
    if not teacher_path or not os.path.isfile(teacher_path):
        raise FileNotFoundError("Set MODEL.TEACHER_WEIGHT to the Night600 teacher checkpoint")
    teacher = copy.deepcopy(_model_core(student))
    checkpoint = torch.load(teacher_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break
    state = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in checkpoint.items()
    }
    incompatible = teacher.load_state_dict(state, strict=False)
    allowed_missing = {
        "base.blocks.11.attn.raw_alpha",
        "base.blocks.11.attn.raw_beta",
    }
    bad_missing = [
        key for key in incompatible.missing_keys if key not in allowed_missing
    ]
    bad_unexpected = [
        key for key in incompatible.unexpected_keys
        if "qnorm_cross_attn" not in key
    ]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Teacher checkpoint is incompatible with the baseline model: "
            "missing={}, unexpected={}".format(bad_missing, bad_unexpected)
        )
    teacher.to(device).requires_grad_(False).eval()
    return teacher


def build_qnorm_mask(teacher_qnorm, mask_ratio):
    """Return detached [B, N_tokens] Top-K mask and head-mean Q-Norm."""
    if teacher_qnorm.ndim != 3:
        raise ValueError("Teacher Q-Norm must have shape [B, H, N_tokens]")
    if teacher_qnorm.requires_grad:
        raise RuntimeError("Teacher Q-Norm must be detached")
    if not 0.0 < mask_ratio <= 1.0:
        raise ValueError("MODEL.QNORM_MASK_RATIO must be in (0, 1]")
    teacher_qnorm_mean = teacher_qnorm.detach().float().mean(dim=1)
    patch_qnorm = teacher_qnorm_mean[:, 1:]
    patch_count = patch_qnorm.shape[-1]
    k = max(1, int(patch_count * mask_ratio))
    top_indices = patch_qnorm.topk(k, dim=-1, largest=True).indices
    patch_mask = torch.zeros_like(patch_qnorm)
    patch_mask.scatter_(1, top_indices, 1.0)
    cls_mask = torch.ones(
        patch_mask.shape[0], 1, dtype=patch_mask.dtype, device=patch_mask.device
    )
    qnorm_mask = torch.cat((cls_mask, patch_mask), dim=1).detach()
    return qnorm_mask, teacher_qnorm_mean.detach()


@torch.no_grad()
def _teacher_mask(cfg, teacher, images, cams, views):
    teacher.eval()
    _, teacher_aux = teacher(
        images,
        cam_label=cams,
        view_label=views,
        qnorm_mask=None,
        return_aux=True,
    )
    teacher_qnorm = teacher_aux["qnorm"].detach()
    qnorm_mask, teacher_qnorm_mean = build_qnorm_mask(
        teacher_qnorm, float(cfg.MODEL.QNORM_MASK_RATIO)
    )
    return qnorm_mask, teacher_qnorm_mean


@torch.no_grad()
def _save_qnorm_sfm_visualizations(
    cfg, epoch, teacher, student, val_loader, device
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset = val_loader.dataset
    if len(dataset) == 0:
        return
    count = min(6, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num=count, dtype=int).tolist()
    samples = [dataset[index] for index in indices]
    images = torch.stack([sample[0] for sample in samples]).to(device)
    cams = torch.tensor([sample[2] for sample in samples], device=device)
    views = torch.tensor([sample[3] for sample in samples], device=device)

    student = _model_core(student)
    was_training = student.training
    student.eval()
    try:
        qnorm_mask, teacher_qnorm_mean = _teacher_mask(
            cfg, teacher, images, cams, views
        )
        _, student_aux = student(
            images,
            cam_label=cams,
            view_label=views,
            qnorm_mask=qnorm_mask,
            return_aux=True,
        )
    finally:
        student.train(was_training)

    before = student_aux["cls_attn_before"].float().mean(dim=1).cpu()
    after = student_aux["cls_attn_after"].float().mean(dim=1).cpu()
    teacher_maps = teacher_qnorm_mean[:, 1:].float().cpu()
    patch_masks = qnorm_mask[:, 1:].float().cpu()
    grid_h = int(student.base.patch_embed.num_y)
    grid_w = int(student.base.patch_embed.num_x)
    if grid_h * grid_w != teacher_maps.shape[-1]:
        raise RuntimeError("Visualization grid does not match the patch count")

    output_dir = Path(cfg.OUTPUT_DIR) / "qnorm_sfm_vis" / "epoch_{:03d}".format(epoch)
    output_dir.mkdir(parents=True, exist_ok=True)
    mean_cpu = torch.tensor(cfg.INPUT.PIXEL_MEAN).view(3, 1, 1)
    std_cpu = torch.tensor(cfg.INPUT.PIXEL_STD).view(3, 1, 1)
    for position, (dataset_index, image) in enumerate(zip(indices, images.cpu()), start=1):
        rgb = (image * std_cpu + mean_cpu).clamp(0, 1).permute(1, 2, 0).numpy()
        image_h, image_w = rgb.shape[:2]
        qnorm_map = teacher_maps[position - 1].reshape(grid_h, grid_w)
        binary_map = patch_masks[position - 1].reshape(grid_h, grid_w)
        before_map = before[position - 1].reshape(grid_h, grid_w)
        after_map = after[position - 1].reshape(grid_h, grid_w)
        qnorm_up = F.interpolate(
            qnorm_map[None, None], size=(image_h, image_w),
            mode="bilinear", align_corners=False
        )[0, 0].numpy()
        mask_up = F.interpolate(
            binary_map[None, None], size=(image_h, image_w), mode="nearest"
        )[0, 0].numpy()
        before_up = F.interpolate(
            before_map[None, None], size=(image_h, image_w),
            mode="bilinear", align_corners=False
        )[0, 0].numpy()
        after_up = F.interpolate(
            after_map[None, None], size=(image_h, image_w),
            mode="bilinear", align_corners=False
        )[0, 0].numpy()
        attention_min = min(float(before_map.min()), float(after_map.min()))
        attention_max = max(float(before_map.max()), float(after_map.max()))

        fig, axes = plt.subplots(1, 5, figsize=(15, 6))
        axes[0].imshow(rgb)
        axes[0].set_title("Original Image")
        for axis, values, title, vmin, vmax, cmap, alpha in (
            (axes[1], qnorm_up, "Teacher Q-Norm", float(qnorm_map.min()),
             float(qnorm_map.max()), "jet", 0.45),
            (axes[2], mask_up, "Q-Norm Binary Mask", 0.0, 1.0, "jet", 0.45),
            (axes[3], before_up, "Student Block12 CLS Before", attention_min,
             attention_max, "jet", 0.45),
            (axes[4], after_up, "Student Block12 CLS After", attention_min,
             attention_max, "jet", 0.45),
        ):
            axis.imshow(rgb)
            axis.imshow(values, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
            axis.set_title(title)
        for axis in axes:
            axis.set_box_aspect(image_h / image_w)
            axis.axis("off")
        fig.suptitle(
            "QNorm-SFM epoch {} sample index {}".format(epoch, dataset_index)
        )
        fig.subplots_adjust(top=0.86, wspace=0.18)
        fig.savefig(
            output_dir / "sample_{:02d}_idx_{:05d}.png".format(
                position, dataset_index
            ),
            dpi=170,
            bbox_inches="tight",
        )
        plt.close(fig)


@torch.no_grad()
def _evaluate(cfg, model, teacher, val_loader, evaluator, device):
    student = _model_core(model)
    was_training = student.training
    student.eval()
    evaluator.reset()
    for img, vid, camid, camids, target_view, _ in val_loader:
        img = img.to(device)
        camids_device = camids.to(device)
        target_view = target_view.to(device)
        qnorm_mask, _ = _teacher_mask(
            cfg, teacher, img, camids_device, target_view
        )
        feat = student(
            img,
            cam_label=camids_device,
            view_label=target_view,
            qnorm_mask=qnorm_mask,
        )
        evaluator.update((feat, vid, camid))
    result = evaluator.compute()
    student.train(was_training)
    return result


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
    vis_period = int(cfg.MODEL.QNORM_SFM_VIS_PERIOD)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info("start QNorm-SFM training")
    model.to(device)
    teacher = _load_teacher(cfg, model, device)

    alpha_names = [
        name for name, parameter in _model_core(model).named_parameters()
        if parameter.requires_grad and name.endswith(("raw_alpha", "raw_beta"))
    ]
    if sorted(alpha_names) != [
        "base.blocks.11.attn.raw_alpha", "base.blocks.11.attn.raw_beta"
    ]:
        raise RuntimeError("Only Block12 raw_alpha/raw_beta may be trainable SFM parameters")
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    named_parameters = dict(_model_core(model).named_parameters())
    for name in alpha_names:
        if id(named_parameters[name]) not in optimizer_parameter_ids:
            raise RuntimeError("{} is missing from the optimizer".format(name))
    logger.info("Optimizer includes learnable QNorm-SFM parameters: %s", alpha_names)

    if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
        logger.info("Using %d GPUs for training", torch.cuda.device_count())
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )

    meters = {
        name: AverageMeter() for name in (
            "loss", "acc", "fg", "rel", "alpha", "beta",
            "entropy_before", "entropy_after", "delta"
        )
    }
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler(enabled=device.type == "cuda")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()
        teacher.eval()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            with torch.no_grad():
                qnorm_mask, _ = _teacher_mask(
                    cfg, teacher, img, target_cam, target_view
                )
            if qnorm_mask.requires_grad:
                raise RuntimeError("Q-Norm mask must be detached")

            with amp.autocast(enabled=device.type == "cuda"):
                student_output, student_aux = model(
                    img,
                    target,
                    cam_label=target_cam,
                    view_label=target_view,
                    qnorm_mask=qnorm_mask,
                    return_aux=True,
                )
                score, feat = student_output
                loss_reid = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss_reid).backward()
            scaler.step(optimizer)
            scaler.update()

            if "center" in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1.0 / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()
            patch_fg_ratio = qnorm_mask[:, 1:].float().mean()
            batch_size = img.shape[0]
            meters["loss"].update(loss_reid.item(), batch_size)
            meters["acc"].update(acc.item(), 1)
            meters["fg"].update(patch_fg_ratio.item(), batch_size)
            meters["rel"].update(
                student_aux["relation_positive_ratio"].detach().item(), batch_size
            )
            meters["alpha"].update(student_aux["alpha"].detach().item(), batch_size)
            meters["beta"].update(student_aux["beta"].detach().item(), batch_size)
            meters["entropy_before"].update(
                student_aux["attn_entropy_before"].detach().item(), batch_size
            )
            meters["entropy_after"].update(
                student_aux["attn_entropy_after"].detach().item(), batch_size
            )
            meters["delta"].update(
                student_aux["attn_delta"].detach().item(), batch_size
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            is_main = not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0
            if is_main and (n_iter + 1) % log_period == 0:
                base_lr = (
                    scheduler._get_lr(epoch)[0]
                    if cfg.SOLVER.WARMUP_METHOD == "cosine"
                    else scheduler.get_lr()[0]
                )
                logger.info(
                    "Epoch[%d] Iter[%d/%d] loss_reid: %.4f alpha: %.4f beta: %.4f "
                    "FG_Ratio: %.4f Rel_Pos: %.4f AttnEnt_Before: %.4f "
                    "AttnEnt_After: %.4f AttnDelta: %.4f Acc: %.4f Base Lr: %.2e",
                    epoch, n_iter + 1, len(train_loader), meters["loss"].avg,
                    meters["alpha"].avg, meters["beta"].avg, meters["fg"].avg,
                    meters["rel"].avg, meters["entropy_before"].avg,
                    meters["entropy_after"].avg, meters["delta"].avg,
                    meters["acc"].avg, base_lr,
                )

        elapsed = time.time() - start_time
        if cfg.SOLVER.WARMUP_METHOD == "cosine":
            scheduler.step(epoch)
        else:
            scheduler.step()
        is_main = not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0
        if is_main:
            logger.info(
                "Epoch %d done. Time per epoch: %.3f[s] Speed: %.1f[samples/s]",
                epoch, elapsed, train_loader.batch_size / (elapsed / (n_iter + 1)),
            )

        if is_main and checkpoint_period > 0 and epoch % checkpoint_period == 0:
            torch.save(
                _model_core(model).state_dict(),
                os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch)),
            )

        if is_main and vis_period > 0 and epoch % vis_period == 0:
            _save_qnorm_sfm_visualizations(
                cfg, epoch, teacher, model, val_loader, device
            )

        if is_main and eval_period > 0 and epoch % eval_period == 0:
            cmc, mAP, _, _, _, _, _ = _evaluate(
                cfg, model, teacher, val_loader, evaluator, device
            )
            logger.info("Validation Results - Epoch: %d", epoch)
            logger.info("mAP: %.1f%%", mAP * 100.0)
            for rank in (1, 5, 10):
                logger.info("CMC curve, Rank-%-3d:%.1f%%", rank, cmc[rank - 1] * 100.0)
            if device.type == "cuda":
                torch.cuda.empty_cache()


def do_inference(cfg, model, val_loader, num_query):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger("transreid.test")
    logger.info("Enter QNorm-SFM inferencing")
    model.to(device)
    teacher = _load_teacher(cfg, model, device)
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    cmc, mAP, _, _, _, _, _ = _evaluate(
        cfg, model, teacher, val_loader, evaluator, device
    )
    logger.info("Validation Results")
    logger.info("mAP: %.1f%%", mAP * 100.0)
    for rank in (1, 5, 10):
        logger.info("CMC curve, Rank-%-3d:%.1f%%", rank, cmc[rank - 1] * 100.0)
    return cmc[0], cmc[4]
