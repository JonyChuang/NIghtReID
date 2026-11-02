import logging
import os
import time
import torch
import torch.nn as nn
import copy
import random
import torch.nn.functional as F
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from pathlib import Path
from qnorm_monitor import QNormMonitor
from utils.qnorm_visualizer import visualize_qnorm_sample

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

    teacher_model = None

    if cfg.MODEL.QNORM_GUIDE:
        teacher_model = copy.deepcopy(model)
        teacher_model.load_param(cfg.MODEL.TEACHER_PATH)

        for param in teacher_model.parameters():
            param.requires_grad = False

        teacher_model.eval()

    # Student 放到 GPU
    if device:
        model.to(local_rank)

        # Teacher 放到 GPU，但不需要 DDP
        if teacher_model is not None:
            teacher_model.to(local_rank)
            teacher_model.eval()

        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], find_unused_parameters=True
        )

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

        core_model = model.module if hasattr(model, 'module') else model
        is_main = not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0

        # 每个 epoch 随机选择 6 个 batch，每个 batch 可视化 1 张
        num_vis = min(6, len(train_loader))
        vis_iters = set(random.sample(range(len(train_loader)), num_vis)) if is_main else set()

        vis_dir = os.path.join(cfg.OUTPUT_DIR, 'qnorm_vis', 'epoch_{:03d}'.format(epoch))
        if is_main:
            os.makedirs(vis_dir, exist_ok=True)

        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            
            qnorm_prior = None
            do_vis = teacher_model is not None and is_main and n_iter in vis_iters

            # 只有被选中的 6 个 iteration 才记录中间特征
            if do_vis:
                core_model.base.blocks[-2].attn.visualize = True
                core_model.base.blocks[-1].attn.visualize = True

            if teacher_model is not None:
                with torch.no_grad(), amp.autocast(enabled=True):
                    qnorm_prior = teacher_model.base.extract_final_qnorm(
                        img, camera_id=target_cam, view_id=target_view
                    )
            with amp.autocast(enabled=True):
                score, feat = model(img,target,cam_label=target_cam,view_label=target_view,qnorm_prior=qnorm_prior)
                loss = loss_fn(score,feat,target,target_cam,)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()


            '''if do_vis:
                try:
                    block11_vis = core_model.base.blocks[-2].attn.vis_data
                    block12_vis = core_model.base.blocks[-1].attn.vis_data

                    # 当前 batch 随机挑一张
                    vis_idx = random.randrange(img.size(0))

                    teacher_map = qnorm_prior[vis_idx].detach().cpu()
                    student_before = block11_vis['qnorm_before'][vis_idx]
                    prior_gain = block11_vis['prior_gain'][vis_idx]
                    student_after = block11_vis['qnorm_after'][vis_idx]
                    cls_attn = block12_vis['cls_attn'][vis_idx]

                    # 反归一化原始图像
                    mean = torch.tensor(
                        cfg.INPUT.PIXEL_MEAN, device=img.device, dtype=img.dtype
                    ).view(3, 1, 1)

                    std = torch.tensor(
                        cfg.INPUT.PIXEL_STD, device=img.device, dtype=img.dtype
                    ).view(3, 1, 1)

                    vis_img = (img[vis_idx] * std + mean).clamp(0, 1)
                    vis_img = vis_img.permute(1, 2, 0).detach().cpu().numpy()

                    grid_h = core_model.base.patch_embed.num_y
                    grid_w = core_model.base.patch_embed.num_x

                    save_path = os.path.join(
                        vis_dir,
                        'iter_{:04d}_idx_{:02d}.png'.format(n_iter, vis_idx)
                    )

                    visualize_qnorm_sample(
                        vis_img,
                        teacher_map,
                        student_before,
                        prior_gain,
                        student_after,
                        cls_attn,
                        grid_h,
                        grid_w,
                        save_path
                    )

                    logger.info(
                        'Saved Q-Norm visualization: epoch {}, iter {}, sample {}'.format(
                            epoch, n_iter, vis_idx
                        )
                    )

                except Exception as e:
                    logger.warning(
                        'Q-Norm visualization failed at epoch {}, iter {}: {}'.format(
                            epoch, n_iter, e
                        )
                    )

                finally:
                    core_model.base.blocks[-2].attn.visualize = False
                    core_model.base.blocks[-1].attn.visualize = False'''

            if do_vis:
                was_training = core_model.training
                block12_attn = core_model.base.blocks[-1].attn

                try:
                    # 当前 batch 随机选择一张图片
                    vis_idx = random.randrange(img.size(0))

                    vis_input = img[vis_idx:vis_idx + 1]
                    vis_cam = target_cam[vis_idx:vis_idx + 1]
                    vis_view = target_view[vis_idx:vis_idx + 1]
                    vis_prior = qnorm_prior[vis_idx:vis_idx + 1]

                    # 可视化时切换 eval，避免 Dropout / DropPath 干扰
                    core_model.eval()
                    block12_attn.visualize = True

                    with torch.no_grad(), amp.autocast(enabled=True):

                        # =====================================================
                        # 1. Plain Student：不使用 Teacher Q-Norm
                        # =====================================================
                        _ = core_model(
                            vis_input,
                            label=None,
                            cam_label=vis_cam,
                            view_label=vis_view,
                            qnorm_prior=None
                        )

                        # 第二次 forward 会覆盖 vis_data，所以这里必须 clone
                        student_qnorm = (
                            block12_attn.vis_data['student_qnorm'][0].clone()
                        )

                        cls_attn_without = (
                            block12_attn.vis_data['cls_attn'][0].clone()
                        )

                        # =====================================================
                        # 2. Guided Student：使用 Teacher Q-Norm
                        # =====================================================
                        _ = core_model(
                            vis_input,
                            label=None,
                            cam_label=vis_cam,
                            view_label=vis_view,
                            qnorm_prior=vis_prior
                        )

                        qnorm_deficit = (
                            block12_attn.vis_data['qnorm_deficit'][0].clone()
                        )

                        cls_attn_with = (
                            block12_attn.vis_data['cls_attn'][0].clone()
                        )

                    # Teacher Q-Norm
                    teacher_qnorm = vis_prior[0].detach().cpu()

                    # =========================================================
                    # CLS Attention Difference
                    #
                    # > 0 : Guidance 后 CLS 更关注该位置
                    # < 0 : Guidance 后 CLS 减少关注该位置
                    # =========================================================
                    cls_attn_diff = cls_attn_with - cls_attn_without

                    # 统计 attention 实际变化量
                    attn_mae = cls_attn_diff.abs().mean().item()

                    attn_cos = F.cosine_similarity(
                        cls_attn_without.unsqueeze(0),
                        cls_attn_with.unsqueeze(0),
                        dim=1
                    ).item()

                    # =========================================================
                    # 反归一化原始图片
                    # =========================================================
                    mean = torch.tensor(
                        cfg.INPUT.PIXEL_MEAN,
                        device=img.device,
                        dtype=img.dtype
                    ).view(3, 1, 1)

                    std = torch.tensor(
                        cfg.INPUT.PIXEL_STD,
                        device=img.device,
                        dtype=img.dtype
                    ).view(3, 1, 1)

                    vis_img = (vis_input[0] * std + mean).clamp(0, 1)
                    vis_img = vis_img.permute(1, 2, 0).detach().cpu().numpy()

                    grid_h = core_model.base.patch_embed.num_y
                    grid_w = core_model.base.patch_embed.num_x

                    save_path = os.path.join(
                        vis_dir,
                        'iter_{:04d}_idx_{:02d}.png'.format(
                            n_iter, vis_idx
                        )
                    )

                    visualize_qnorm_sample(
                        vis_img,
                        teacher_qnorm,
                        student_qnorm,
                        qnorm_deficit,
                        cls_attn_without,
                        cls_attn_with,
                        cls_attn_diff,
                        grid_h,
                        grid_w,
                        save_path
                    )

                    logger.info(
                        "Q-Vis Epoch {} Iter {} | "
                        "AttnCos: {:.6f}, AttnMAE: {:.6f}".format(
                            epoch,
                            n_iter,
                            attn_cos,
                            attn_mae
                        )
                    )

                except Exception as e:
                    logger.warning(
                        "Q-Norm visualization failed at epoch {}, iter {}: {}".format(
                            epoch, n_iter, e
                        )
                    )

                finally:
                    block12_attn.visualize = False

                    if was_training:
                        core_model.train()

                        
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
                    core_model = model.module if hasattr(model, 'module') else model

                    #qnorm_logit = core_model.base.qnorm_mix_logit.item()
                    #qnorm_mix = torch.sigmoid(core_model.base.qnorm_mix_logit).item()
                    qnorm_logit = core_model.base.qnorm_alpha_logit.item()
                    qnorm_alpha = torch.sigmoid(core_model.base.qnorm_alpha_logit).item()

                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}, Q-Norm Alpha: {:.3f},"
                                .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr,qnorm_alpha))
        
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
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        #feat = model(img, cam_label=camids, view_label=target_view)
                        with torch.no_grad():
                            qnorm_prior = teacher_model.base.extract_final_qnorm(
                                img, camera_id=camids, view_id=target_view
                            )

                        feat = model(img, cam_label=camids, view_label=target_view, qnorm_prior=qnorm_prior)
                        evaluator.update((feat, vid, camid))
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


