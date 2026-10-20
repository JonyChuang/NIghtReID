import logging
import os
import time
import torch
import torch.nn as nn
import copy
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from pathlib import Path
from qnorm_monitor import QNormMonitor
from visualize_qnorm import save_context_visualizations

def build_teacher(cfg, model):
    student = model.module if hasattr(model, "module") else model
    teacher = copy.deepcopy(student)

    teacher.to(next(student.parameters()).device)

    if not cfg.MODEL.TEACHER_WEIGHT:
        raise ValueError("cfg.MODEL.TEACHER_WEIGHT is empty")

  
    teacher.load_param(
        cfg.MODEL.TEACHER_WEIGHT
    )

    teacher.requires_grad_(False)
    teacher.eval()
    return teacher

def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module

    return model

@torch.no_grad()
def get_teacher_qnorm(
        teacher,
        images,
        camera_ids,
        view_ids,
        prefix_tokens=1,
):
    teacher_core = unwrap_model(teacher)

    _, qnorm = teacher_core.base.forward_features(
        images,
        camera_id=camera_ids,
        view_id=view_ids,
        return_tokens=True,
        return_qnorm=True,
    )

    if qnorm.dim() != 2:
        raise RuntimeError(
            "Expected Q-Norm with shape [B, L], "
            f"but got {tuple(qnorm.shape)}"
        )

    num_patches = (
        teacher_core.base.patch_embed.num_patches
    )

    if qnorm.size(1) == num_patches + prefix_tokens:
        qnorm = qnorm[:, prefix_tokens:]
    elif qnorm.size(1) != num_patches:
        raise RuntimeError(
            "Unexpected Q-Norm token count: "
            f"{qnorm.size(1)}; expected "
            f"{num_patches} or "
            f"{num_patches + prefix_tokens}"
        )

    return qnorm.detach().float()

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
    teacher_model = build_teacher(cfg, model)

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()


    '''qnorm_monitor = QNormMonitor(
        model,
        val_loader,  # 推荐使用验证集，避免随机擦除等训练增强
        Path(cfg.OUTPUT_DIR) / "qnorm_vis",
        every=10,
        num_images=6,
        input_height=cfg.INPUT.SIZE_TRAIN[0],
        input_width=cfg.INPUT.SIZE_TRAIN[1],
        patch_size=cfg.MODEL.STRIDE_SIZE[0],
        pixel_mean=cfg.INPUT.PIXEL_MEAN,
        pixel_std=cfg.INPUT.PIXEL_STD,
        device=device,
    )'''


    '''model_for_log = (
    model.module
    if hasattr(model, "module")
    else model
    )

    qnorm_pooling = getattr(
        model_for_log,
        "qnorm_guided_pooling",
        None,
    )'''

    model_core = (
    model.module
    if hasattr(model, "module")
        else model
    )

    gate = getattr(
        model_core.base,
        "last_qnorm_gate",
        None,
    )
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        model.train()
        teacher_model.eval()
        teacher_model.requires_grad_(False)

        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            teacher_qnorm = get_teacher_qnorm(teacher_model,img,target_cam,target_view,
                prefix_tokens=(
                    cfg.MODEL.QNORM_PREFIX_TOKENS
                ),
            )

            with amp.autocast(enabled=True):
                score, feat = model(img,target,cam_label=target_cam,view_label=target_view,teacher_qnorm=teacher_qnorm,)
                loss = loss_fn(score,feat,target,target_cam,)

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
                        logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))
            else:
                if (n_iter + 1) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}, Gate: {:.3f}"
                                .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr, gate.mean().item() if gate is not None else 0.0))
        
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()
        ''' try:
            qnorm_monitor.on_epoch_end(epoch)
        except Exception:
            logger.exception(
                "Q-Norm visualization failed at epoch %d", epoch
            )'''

        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter + 1), train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
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
                            # 正常评估仍然使用真实 Q-Norm 的特征
                            feat = feat_real
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                evaluator.reset()
                eval_model = model.module if hasattr(model, "module") else model
                is_main_process = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        #feat = model(img, cam_label=camids, view_label=target_view)
                        teacher_qnorm = get_teacher_qnorm(
                                    teacher_model,
                                    img,
                                    camids,
                                    target_view,
                                    prefix_tokens=cfg.MODEL.QNORM_PREFIX_TOKENS,
                                )

                        capture_maps = epoch % 10 == 0 and n_iter == 0 and is_main_process

                        feat = eval_model(
                                    img,
                                    cam_label=camids,
                                    view_label=target_view,
                                    teacher_qnorm=teacher_qnorm,
                                    return_activation_maps=capture_maps,
                                )

                        evaluator.update((feat, vid, camid))

                        if capture_maps:
                            maps = eval_model.qnorm_guided_pooling.get_visualization_maps()
                            save_context_visualizations(
                                images=img,
                                maps=maps,
                                output_dir=Path(cfg.OUTPUT_DIR) / "qnorm_context_vis",
                                epoch=epoch,
                                grid_size=(16, 8),
                                max_images=6,
                                pixel_mean=cfg.INPUT.PIXEL_MEAN,
                                pixel_std=cfg.INPUT.PIXEL_STD,
                            )
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()
    qnorm_monitor.close()

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


