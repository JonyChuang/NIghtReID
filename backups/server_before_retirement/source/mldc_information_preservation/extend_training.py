"""Continue a completed shared-CNL checkpoint with its model and image memory."""
import argparse
import json
import math
import os
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ConstantContinuationLR:
    """Keep every optimizer group at the explicitly selected continuation LR."""
    def __init__(self, optimizer, learning_rate):
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('Continuation learning rate must be finite and positive')
        self.optimizer = optimizer
        self.learning_rate = float(learning_rate)
        self.step()

    def step(self, epoch=None):
        for group in self.optimizer.param_groups:
            group['lr'] = self.learning_rate

    def state_dict(self):
        return dict(kind='constant_continuation', learning_rate=self.learning_rate)

    def load_state_dict(self, state):
        if state != self.state_dict():
            raise ValueError('Continuation learning rate differs from saved schedule')
        self.step()


def continuation_config(saved, checkpoint_path, extra_epochs, output, device_id=None):
    from yacs.config import CfgNode
    if type(extra_epochs) is not int or extra_epochs <= 0:
        raise ValueError('extra_epochs must be a positive integer')
    cfg = CfgNode.load_cfg(saved['config'])
    if cfg.SOLVER.AUXILIARY_LOSS != 'shared_cnl' or not saved.get('shared_cnl_memory'):
        raise ValueError('Continuation requires the shared-CNL checkpoint and complete image memory')
    if saved.get('shared_cnl_spec', {}).get('memory_schema') != 'image_index_fused_and_four_layers_v2':
        raise ValueError('Unsupported continuation memory schema')
    if int(saved['epoch']) != cfg.SOLVER.MAX_EPOCHS:
        raise ValueError('This entry extends a completed run; select its final checkpoint')
    cfg.MODEL.PRETRAIN_CHOICE = 'self'
    cfg.MODEL.PRETRAIN_PATH = str(checkpoint_path)
    cfg.SOLVER.MAX_EPOCHS = int(saved['epoch']) + extra_epochs
    cfg.OUTPUT_DIR = str(output)
    if device_id is not None:
        cfg.MODEL.DEVICE_ID = device_id
    cfg.freeze()
    return cfg


def restore_memory(cfg, model, dataset, saved, device):
    from .shared_cnl import create_shared_memory
    if dataset.fingerprint != saved['training_dataset_fingerprint']:
        raise ValueError('Training image indices differ from the saved memory')
    memory = create_shared_memory(cfg, model, dataset, device)
    if memory is None:
        raise ValueError('Continuation requires nonzero configured CNL weight')
    import torch
    if not torch.equal(memory.cameras.cpu(), saved['shared_cnl_memory']['cameras'].cpu()):
        raise ValueError('Training camera indices differ from the saved memory')
    memory.load_state_dict(saved['shared_cnl_memory'], strict=True)
    return memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--extra_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=8e-7)
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--device_id')
    parser.add_argument('--verify_only', action='store_true', help='CPU restore check without output creation or training')
    args = parser.parse_args()
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('Continuation learning rate must be finite and positive')
    import numpy as np
    import torch
    saved = torch.load(str(args.checkpoint),map_location='cpu',mmap=True,weights_only=False)
    start = int(saved['epoch'])
    output = args.output_dir or args.checkpoint.parent.with_name(args.checkpoint.parent.name + '_extend%d' % (start + args.extra_epochs))
    if output.resolve() == args.checkpoint.parent.resolve() or (output.exists() and any(output.iterdir())):
        raise FileExistsError('Use an empty continuation directory separate from the original run')
    cfg = continuation_config(saved,args.checkpoint,args.extra_epochs,output,args.device_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    if not args.verify_only and (cfg.MODEL.DEVICE != 'cuda' or not torch.cuda.is_available()):
        raise RuntimeError('Continuation requires a visible CUDA GPU')
    random.seed(cfg.SOLVER.SEED);np.random.seed(cfg.SOLVER.SEED)
    torch.manual_seed(cfg.SOLVER.SEED);torch.cuda.manual_seed_all(cfg.SOLVER.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    from datasets import make_dataloader
    from solver import make_optimizer
    from utils.logger import setup_logger
    from .indexed_data import indexed_loader
    from .model.make_model_mldc import make_model,configure_head_learning_rates
    from .loss.make_loss_mldc import make_loss
    from .processor.processor_mldc import do_train
    loader,_,val,nq,nc,cameras,views = make_dataloader(cfg)
    loader,dataset = indexed_loader(loader)
    if dataset.fingerprint != saved['training_dataset_fingerprint']:
        raise ValueError('Training dataset fingerprint changed')
    model = make_model(cfg,nc,cameras,views)
    device = 'cpu' if args.verify_only else 'cuda:0'
    model.to(device)
    memory = restore_memory(cfg,model,dataset,saved,device)
    loss_fn,center = make_loss(cfg,nc)
    optimizer,_ = make_optimizer(cfg,model,center)
    configure_head_learning_rates(cfg,model,optimizer)
    optimizer_restored = 'optimizer_state' in saved
    if optimizer_restored:
        optimizer.load_state_dict(saved['optimizer_state'])
    scheduler = ConstantContinuationLR(optimizer,args.learning_rate)
    scaler_state = saved.get('scaler_state')
    source_metrics = saved.get('metrics') or {}
    if args.verify_only:
        from .cnl_schedule import auxiliary_weight_for_epoch
        print(json.dumps(dict(source_epoch=start,first_epoch=start+1,final_epoch=cfg.SOLVER.MAX_EPOCHS,
            source_metrics=source_metrics,learning_rates=sorted(set(g['lr'] for g in optimizer.param_groups)),
            memory_images=int(memory.valid.sum()),dataset_fingerprint_matches=True,
            optimizer_state_restored=optimizer_restored,cnl_effective_weight=auxiliary_weight_for_epoch(cfg,start+1),
            optimizer_steps=0,training_started=False,output_created=False)),flush=True)
        return
    output.mkdir(parents=True,exist_ok=True)
    logger = setup_logger('transreid',str(output),if_train=True)
    metadata = dict(variant=saved['variant'],config=cfg.dump(),continuation=True,
        source_checkpoint=str(args.checkpoint.resolve()),source_epoch=start,
        final_epoch=cfg.SOLVER.MAX_EPOCHS,extra_epochs=args.extra_epochs,
        learning_rate=args.learning_rate,learning_rate_policy='constant in every parameter group; no new LR warmup',
        optimizer_state_restored=optimizer_restored,optimizer_momentum_reset=not optimizer_restored,
        scaler_state_restored=bool(scaler_state),rng_state_restored=False,
        source_metrics=source_metrics,training_dataset_fingerprint=dataset.fingerprint,
        memory_images_restored=int(memory.valid.sum()),memory_reinitialized=False,
        cnl_weight=cfg.SOLVER.CNL_WEIGHT,cnl_warmup_repeated=False,
        seed=cfg.SOLVER.SEED,note='Model/BN/fusion/memory continuation; legacy checkpoint lacks optimizer and RNG state')
    (output/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    logger.info('Continue epoch %d through %d, LR %.9g, CNL weight %.3f, restored memory %d',
        start+1,cfg.SOLVER.MAX_EPOCHS,args.learning_rate,cfg.SOLVER.CNL_WEIGHT,int(memory.valid.sum()))
    logger.info('Optimizer state restored: %s; AMP state restored: %s; RNG state restarted from seed',optimizer_restored,bool(scaler_state))
    do_train(cfg,model,loader,val,optimizer,scheduler,loss_fn,nq,dataset,start_epoch=start,
        memory=memory,scaler_state=scaler_state,initial_best_map=source_metrics.get('mAP',-1.))


if __name__ == '__main__':
    main()
