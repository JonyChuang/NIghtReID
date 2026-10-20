import logging
import os
import time
import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist


def _model_core(model):
    return model.module if hasattr(model, "module") else model


def _load_teacher(cfg, student, device):
    """Load the Night600 teacher only into an independent, frozen model."""
    teacher_path = cfg.MODEL.TEACHER_WEIGHT
    if not teacher_path or not os.path.isfile(teacher_path):
        raise FileNotFoundError("Set MODEL.TEACHER_WEIGHT to a trained teacher checkpoint")
    teacher = copy.deepcopy(_model_core(student))
    checkpoint = torch.load(teacher_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
    }
    incompatible = teacher.load_state_dict(state, strict=False)
    unexpected = incompatible.unexpected_keys
    if incompatible.missing_keys or any(
        not key.startswith("qnorm_cross_attn.") for key in unexpected
    ):
        raise RuntimeError(
            "Teacher checkpoint is incompatible with the baseline backbone: "
            f"missing={incompatible.missing_keys}, unexpected={unexpected}"
        )
    if unexpected:
        logging.getLogger("transreid.train").warning(
            "Ignored %d unused teacher qnorm_cross_attn parameters; "
            "all baseline backbone parameters loaded.", len(unexpected)
        )
    teacher.to(device).requires_grad_(False).eval()
    return teacher


def qcat_loss_and_metrics(teacher_qnorm, student_cls_attn):
    """Raw head-mean maps; L2 normalization is used only for the loss."""
    if teacher_qnorm.shape != student_cls_attn.shape or teacher_qnorm.ndim != 3:
        raise ValueError("QCAT tensors must have matching [B, H, N] shapes")
    if teacher_qnorm.requires_grad:
        raise RuntimeError("The teacher Q-Norm target must be detached")
    if not student_cls_attn.requires_grad:
        raise RuntimeError("Student CLS attention must retain its gradient")

    teacher_map = teacher_qnorm.detach().float().mean(dim=1)
    student_map = student_cls_attn.float().mean(dim=1)
    teacher_map_norm = F.normalize(teacher_map, p=2, dim=-1)
    student_map_norm = F.normalize(student_map, p=2, dim=-1)
    loss_qcat = ((teacher_map_norm - student_map_norm) ** 2).sum(dim=-1).mean()

    with torch.no_grad():
        qcat_cos = F.cosine_similarity(teacher_map_norm, student_map_norm, dim=-1).mean()
        student_prob = student_map / (student_map.sum(dim=-1, keepdim=True) + 1e-6)
        entropy = -(student_prob * torch.log(student_prob + 1e-8)).sum(dim=-1).mean()
        k = max(1, int(teacher_map.shape[-1] * 0.2))
        idx = teacher_map.topk(k, dim=-1).indices
        top20_mass = student_prob.gather(1, idx).sum(dim=-1).mean()
    return loss_qcat, qcat_cos, entropy, top20_mass


@torch.no_grad()
def _save_qcat_visualizations(cfg, epoch, teacher, initial_student, student, val_loader, device):
    """Compare the same validation tensors under teacher, initial and current student."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset = val_loader.dataset
    if len(dataset) == 0:
        return
    chosen = random.sample(range(len(dataset)), min(6, len(dataset)))
    samples = [dataset[index] for index in chosen]
    images = torch.stack([sample[0] for sample in samples]).to(device)
    cams = torch.tensor([sample[2] for sample in samples], device=device)
    views = torch.tensor([sample[3] for sample in samples], device=device)

    student = _model_core(student)
    was_training = student.training
    student.eval()
    initial_student.to(device).eval()
    try:
        _, teacher_aux = teacher(images, cam_label=cams, view_label=views, return_aux=True)
        _, before_aux = initial_student(images, cam_label=cams, view_label=views, return_aux=True)
        _, after_aux = student(images, cam_label=cams, view_label=views, return_aux=True)
    finally:
        initial_student.cpu()
        student.train(was_training)

    teacher_map = teacher_aux["last_qnorm"].float().mean(dim=1).cpu()
    before_map = before_aux["last_cls_attn"].float().mean(dim=1).cpu()
    after_map = after_aux["last_cls_attn"].float().mean(dim=1).cpu()
    grid_h = student.base.patch_embed.num_y
    grid_w = student.base.patch_embed.num_x
    if teacher_map.shape[-1] != grid_h * grid_w:
        raise RuntimeError("QCAT visualization grid does not match the patch count")

    output_dir = Path(cfg.OUTPUT_DIR) / "qcat_raw_vis"
    output_dir.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor(cfg.INPUT.PIXEL_MEAN).view(3, 1, 1)
    std = torch.tensor(cfg.INPUT.PIXEL_STD).view(3, 1, 1)
    for position, (dataset_index, image) in enumerate(zip(chosen, images.cpu()), start=1):
        rgb = (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        fig, axes = plt.subplots(1, 4, figsize=(14, 6))
        axes[0].imshow(rgb)
        axes[0].set_title("Original Image")
        for axis, values, title in zip(
            axes[1:],
            (teacher_map[position - 1], before_map[position - 1], after_map[position - 1]),
            ("Teacher Q-Norm", "Student CLS Before", "Student CLS After"),
        ):
            display = values.reshape(grid_h, grid_w)
            display = (display - display.min()) / (display.max() - display.min() + 1e-8)
            display = F.interpolate(
                display[None, None], size=rgb.shape[:2], mode="bilinear", align_corners=False
            )[0, 0].numpy()
            axis.imshow(rgb)
            axis.imshow(display, cmap="turbo", alpha=0.55, vmin=0, vmax=1)
            axis.set_title(title)
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"epoch_{epoch:03d}_sample_{position:02d}_idx_{dataset_index}.png", dpi=150)
        plt.close(fig)

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

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    model.to(device)
    # The initial student is the untrained reference for every before/after panel.
    initial_student = copy.deepcopy(model).cpu().requires_grad_(False).eval()
    teacher = _load_teacher(cfg, model, device)
    if cfg.MODEL.DIST_TRAIN:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    reid_meter = AverageMeter()
    qcat_meter = AverageMeter()
    cosine_meter = AverageMeter()
    entropy_meter = AverageMeter()
    top20_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler(enabled=device.type == "cuda")
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        reid_meter.reset()
        qcat_meter.reset()
        cosine_meter.reset()
        entropy_meter.reset()
        top20_meter.reset()
        evaluator.reset()
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
                _, teacher_aux = teacher(
                    img, cam_label=target_cam, view_label=target_view, return_aux=True
                )
                teacher_qnorm = teacher_aux["last_qnorm"].detach()
            assert not teacher_qnorm.requires_grad

            with amp.autocast(enabled=device.type == "cuda"):
                (score, feat), student_aux = model(
                    img, target, cam_label=target_cam,
                    view_label=target_view, return_aux=True,
                )
                student_cls_attn = student_aux["last_cls_attn"]
                assert student_cls_attn.requires_grad
                loss_reid = loss_fn(score, feat, target, target_cam)
                loss_qcat, qcat_cos, entropy, top20_mass = qcat_loss_and_metrics(
                    teacher_qnorm, student_cls_attn
                )
                loss = loss_reid + cfg.SOLVER.QCAT_LOSS_WEIGHT * loss_qcat

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
            reid_meter.update(loss_reid.item(), img.shape[0])
            qcat_meter.update(loss_qcat.item(), img.shape[0])
            cosine_meter.update(qcat_cos.item(), img.shape[0])
            entropy_meter.update(entropy.item(), img.shape[0])
            top20_meter.update(top20_mass.item(), img.shape[0])
            acc_meter.update(acc.item(), 1)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if (not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0) and (n_iter + 1) % log_period == 0:
                base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                logger.info(
                    "Epoch[%d] Iter[%d/%d] loss_reid: %.4f loss_qcat: %.4f "
                    "loss_total: %.4f QCAT cosine: %.4f Student CLS entropy: %.4f "
                    "Top20_Q_Mass: %.4f Acc: %.4f Base Lr: %.2e",
                    epoch, n_iter + 1, len(train_loader), reid_meter.avg,
                    qcat_meter.avg, loss_meter.avg, cosine_meter.avg,
                    entropy_meter.avg, top20_meter.avg, acc_meter.avg, base_lr,
                )

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()
        is_main = not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0
        if is_main:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter + 1), train_loader.batch_size / time_per_batch))

        if is_main and checkpoint_period > 0 and epoch % checkpoint_period == 0:
            torch.save(
                _model_core(model).state_dict(),
                os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)),
            )

        if is_main and eval_period > 0 and epoch % eval_period == 0:
            eval_model = _model_core(model)
            eval_model.eval()
            evaluator.reset()
            for img, vid, camid, camids, target_view, _ in val_loader:
                with torch.no_grad():
                    feat = eval_model(
                        img.to(device), cam_label=camids.to(device),
                        view_label=target_view.to(device),
                    )
                    evaluator.update((feat, vid, camid))
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: %d", epoch)
            logger.info("mAP: %.1f%%", 100.0 * mAP)
            for rank in (1, 5, 10):
                logger.info("CMC curve, Rank-%-3d: %.1f%%", rank, 100.0 * cmc[rank - 1])
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if is_main and epoch % 10 == 0:
            _save_qcat_visualizations(
                cfg, epoch, teacher, initial_student, model, val_loader, device
            )


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


def main():
    """Standalone QCAT-raw entry point; the baseline train.py stays untouched."""
    from config import cfg
    from datasets import make_dataloader
    from loss import make_loss
    from model.make_model_qcat_raw import make_model
    from solver import make_optimizer, WarmupMultiStepLR
    from solver.scheduler_factory import create_scheduler
    from utils.logger import setup_logger

    parser = argparse.ArgumentParser(description="QCAT-raw Night600 training")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.defrost()
    cfg.SOLVER.QCAT_LOSS_WEIGHT = 1.0
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if cfg.MODEL.NAME != "transformer" or cfg.MODEL.JPM:
        raise ValueError("QCAT-raw requires a Transformer with MODEL.JPM=False")
    if cfg.MODEL.TRANSFORMER_TYPE not in (
        "vit_base_patch16_224_TransReID", "deit_base_patch16_224_TransReID",
        "vit_small_patch16_224_TransReID", "deit_small_patch16_224_TransReID",
    ):
        raise ValueError("QCAT-raw supports only the ViT TransReID backbone")
    if cfg.SOLVER.QCAT_LOSS_WEIGHT < 0:
        raise ValueError("SOLVER.QCAT_LOSS_WEIGHT must be non-negative")
    if not cfg.OUTPUT_DIR:
        raise ValueError("Set an experiment-specific OUTPUT_DIR")
    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    random.seed(cfg.SOLVER.SEED)
    np.random.seed(cfg.SOLVER.SEED)
    torch.manual_seed(cfg.SOLVER.SEED)
    torch.cuda.manual_seed_all(cfg.SOLVER.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.MODEL.DEVICE_ID)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger("transreid", cfg.OUTPUT_DIR, if_train=True)
    logger.info("QCAT-raw config:\n%s", cfg)

    train_loader, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    if cfg.SOLVER.WARMUP_METHOD == "cosine":
        scheduler = create_scheduler(cfg, optimizer)
    else:
        scheduler = WarmupMultiStepLR(
            optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
            cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_EPOCHS,
            cfg.SOLVER.WARMUP_METHOD,
        )
    do_train(
        cfg, model, center_criterion, train_loader, val_loader,
        optimizer, optimizer_center, scheduler, loss_func,
        num_query, args.local_rank,
    )


if __name__ == "__main__":
    main()
