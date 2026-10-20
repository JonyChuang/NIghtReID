import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.logger import setup_logger
from datasets import make_dataloader
from ffvt_token_fusion.model.make_model_alf import make_model, add_alf_config, configure_head_learning_rates
from solver import make_optimizer, WarmupMultiStepLR
from solver.scheduler_factory import create_scheduler
from ffvt_token_fusion.loss.make_loss_alf import make_alf_loss
from ffvt_token_fusion.processor.processor_alf import do_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg as base_cfg
cfg = add_alf_config(base_cfg.clone())
import torch.distributed as dist


def make_optimizer_with_ffvt_lr(cfg, model, center_criterion):
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    configure_head_learning_rates(cfg, model, optimizer)
    return optimizer, optimizer_center


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Night600 FFVT dual-branch training")
    parser.add_argument(
        "--config_file", default="ffvt_token_fusion/variants/maws_global_and_fused_tokens/config.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    if not torch.cuda.is_available():
        raise RuntimeError('FFVT end-to-end training requires a CUDA GPU; none is visible to this server')
    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    try:
        os.makedirs(output_dir)
    except:
        pass

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    #  logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            #  logger.info(config_str)

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    logger.info("Running with config:\n{}".format(cfg))


    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    loss_func, center_criterion = make_alf_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer_with_ffvt_lr(cfg, model, center_criterion)
    logger.info('FFVT backbone/fusion weight LR: {:.2e}; new BN/classifier weight LR: {:.2e} ({}x)'.format(
        cfg.SOLVER.BASE_LR, cfg.SOLVER.BASE_LR * cfg.SOLVER.FFVT_HEAD_LR_FACTOR,
        cfg.SOLVER.FFVT_HEAD_LR_FACTOR))

    if cfg.SOLVER.WARMUP_METHOD == 'cosine':
        logger.info('===========using cosine learning rate=======')
        scheduler = create_scheduler(cfg, optimizer)
    else:
        logger.info('===========using normal learning rate=======')
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                      cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, args.local_rank
    )
    #  print(cfg.OUTPUT_DIR)
    #  print(cfg.MODEL.PRETRAIN_PATH)
