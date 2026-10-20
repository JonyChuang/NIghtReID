from yacs.config import CfgNode as CN
from mldc_cls_readout.model.make_model_mldc import add_mldc_config


def add_config(cfg):
    add_mldc_config(cfg)
    cfg.MUFASA = CN()
    cfg.MUFASA.BASELINE = '/root/log/transreid/night600/vit_base_ics_a3pfn_mldc/transformer_120.pth'
    cfg.MUFASA.LAYERS = [9, 10, 11, 12]  # One-based block numbers.
    cfg.MUFASA.NUM_SLOTS = 4
    cfg.MUFASA.SLOT_DIM = 256
    cfg.MUFASA.ITERATIONS = 3
    cfg.MUFASA.SA_HIDDEN = 1024
    cfg.MUFASA.MATCH = True
    cfg.MUFASA.FUSION = 'mfusion'
    cfg.MUFASA.RESIDUAL_SCALE = 0.1
    cfg.MUFASA.RECON_WEIGHT = 0.1
    cfg.MUFASA.DECODER_HIDDEN = 2048
    cfg.MUFASA.VISUALIZE = True
    cfg.MUFASA.VIS_PERIOD = 20
    cfg.MUFASA.VIS_SAMPLES = 6
    cfg.MUFASA.CPU_THREADS = 8
    return cfg


def validate(cfg):
    if cfg.MODEL.DIST_TRAIN or cfg.MODEL.DEVICE != 'cuda':
        raise ValueError('The training entry uses one CUDA GPU')
    if cfg.SOLVER.OPTIMIZER_NAME != 'Adam':
        raise ValueError('This protocol uses Adam for the new modules')
    if cfg.SOLVER.MAX_EPOCHS <= cfg.SOLVER.WARMUP_EPOCHS or cfg.SOLVER.WARMUP_EPOCHS < 1:
        raise ValueError('Require 0 < warmup epochs < total epochs')
    if cfg.SOLVER.IMS_PER_BATCH % cfg.DATALOADER.NUM_INSTANCE or cfg.DATALOADER.NUM_INSTANCE < 2:
        raise ValueError('PK sampling requires equal groups with at least two images')
    if cfg.DATALOADER.SAMPLER != 'softmax_triplet':
        raise ValueError('Use the original PK sampler')
    if cfg.TEST.RE_RANKING or cfg.TEST.FEAT_NORM != 'yes' or cfg.TEST.NECK_FEAT != 'after':
        raise ValueError('Use original BN-after baseline with L2 normalization and no reranking')
    if cfg.MUFASA.RESIDUAL_SCALE < 0 or cfg.MUFASA.RESIDUAL_SCALE >= 1:
        raise ValueError('Residual scale must be in [0,1)')
    if not 1 <= cfg.MUFASA.NUM_SLOTS <= 6:
        raise ValueError('Exact permutation assignment supports 1--6 slots')
    if len(cfg.MUFASA.LAYERS) != 4 or any(l < 1 or l > 12 for l in cfg.MUFASA.LAYERS):
        raise ValueError('Use four input branches with block indices in 1--12')
    if cfg.MUFASA.FUSION not in ('mfusion', 'mean'):
        raise ValueError('Unknown slot fusion')
    if cfg.MODEL.IF_LABELSMOOTH != 'off' or cfg.MODEL.IF_WITH_CENTER != 'no':
        raise ValueError('Use unsmoothed ID loss without center loss')
    if not cfg.MODEL.NO_MARGIN:
        raise ValueError('Use the original soft-margin batch-hard triplet')
