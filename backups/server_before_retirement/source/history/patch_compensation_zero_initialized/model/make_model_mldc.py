"""Frozen deep-four CLS anchor plus trainable multi-layer patch compensation."""
from pathlib import Path
import torch
from torch import nn
from .backbones.vit_pytorch_mldc import vit_base_patch16_224_TransReID
from .mldc import MLDC
from .patch_compensation import PatchCompensation

EXPERIMENT = 'mldc_patch_compensation_20261016'
PACKAGE = Path(__file__).resolve().parents[1]


def add_mldc_config(cfg):
    cfg.MODEL.MLDC_READOUT = 'mldc'
    cfg.MODEL.MLDC_TEMPERATURE = 0.9
    cfg.MODEL.MLDC_LAYERS = [9, 10, 11, 12]
    cfg.MODEL.MLDC_INIT = 'uniform'
    cfg.MODEL.MLDC_TRAIN_FEATURE = 'patch_compensation'
    cfg.MODEL.MLDC_DEEP_INIT_MASS = 0.75
    cfg.MODEL.PATCH_LAYERS = list(range(1, 13))
    cfg.MODEL.PATCH_HIDDEN_DIM = 192
    cfg.MODEL.PATCH_HEADS = 4
    cfg.MODEL.PATCH_USE_NULL = True
    cfg.MODEL.PATCH_ZERO_OUTPUT = True
    cfg.MODEL.PATCH_BASELINE_CHECKPOINT = str(PACKAGE / 'initialization/deep4_uniform_epoch120.pth')
    cfg.SOLVER.MLDC_HEAD_LR_FACTOR = 5.0
    cfg.SOLVER.MLDC_WEIGHT_LR_FACTOR = 1.0
    cfg.TEST.MLDC_VISUALIZE = True
    cfg.TEST.MLDC_VIS_SAMPLES = 6
    cfg.TEST.MLDC_VIS_PERIOD = 10
    cfg.TEST.MLDC_RETRIEVAL = 'mldc'
    return cfg


def configure_head_learning_rates(cfg, model, optimizer):
    # Patch parameters use BASE_LR; the classifier keeps the prior head factor.
    heads = {id(p) for n, p in model.named_parameters()
             if p.requires_grad and n.startswith('classifier.')}
    for group in optimizer.param_groups:
        ids = {id(p) for p in group['params']}
        if ids & heads:
            if not ids <= heads:
                raise RuntimeError('Optimizer mixes classifier and patch parameters')
            group['lr'] *= cfg.SOLVER.MLDC_HEAD_LR_FACTOR


class MLDCReID(nn.Module):
    def __init__(self, cfg, num_classes, camera_num, view_num):
        super().__init__()
        if cfg.MODEL.NAME != 'transformer' or cfg.MODEL.TRANSFORMER_TYPE != 'vit_base_patch16_224_TransReID':
            raise ValueError('Patch compensation requires the baseline Night600 ViT-Base')
        if any((cfg.MODEL.JPM, cfg.MODEL.GEM_POOLING, cfg.MODEL.USE_DCE,
                cfg.MODEL.QNORM_PRIOR, cfg.MODEL.QNORM_GUIDE,
                cfg.MODEL.DIST_TRAIN, cfg.MODEL.REDUCE_FEAT_DIM)):
            raise ValueError('Stage 1 requires the original plain single-GPU backbone')
        if cfg.MODEL.ID_LOSS_TYPE != 'softmax':
            raise ValueError('Use the original softmax identity supervision')
        if (cfg.MODEL.MLDC_READOUT != 'mldc' or list(cfg.MODEL.MLDC_LAYERS) != [9, 10, 11, 12]
                or cfg.MODEL.MLDC_TRAIN_FEATURE != 'patch_compensation'
                or cfg.TEST.MLDC_RETRIEVAL != 'mldc'
                or cfg.TEST.NECK_FEAT not in ('before', 'after')):
            raise ValueError('This copy uses compensated deep-four CLS, without last-CLS addition')
        if cfg.MODEL.PRETRAIN_CHOICE != 'self':
            raise ValueError('Initialize from the completed deep-four checkpoint, not ImageNet')
        if cfg.SOLVER.MLDC_WEIGHT_LR_FACTOR != 1.0:
            raise ValueError('CLS fusion weights are frozen in stage 1')
        self.readout, self.retrieval = 'mldc', 'mldc'
        self.train_feature = 'patch_compensation'
        self.checkpoint_epoch, self.baseline_epoch = 0, 120
        self.baseline_checkpoint = str(cfg.MODEL.PATCH_BASELINE_CHECKPOINT)
        self.mldc = MLDC(12, cfg.MODEL.MLDC_TEMPERATURE,
                         init=cfg.MODEL.MLDC_INIT, selected_layers=cfg.MODEL.MLDC_LAYERS)
        self.base = vit_base_patch16_224_TransReID(
            img_size=cfg.INPUT.SIZE_TRAIN, stride_size=cfg.MODEL.STRIDE_SIZE,
            camera=camera_num if cfg.MODEL.SIE_CAMERA else 0,
            view=view_num if cfg.MODEL.SIE_VIEW else 0,
            sie_xishu=cfg.MODEL.SIE_COE, stem_conv=cfg.MODEL.STEM_CONV,
            drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        dim = self.base.in_planes
        self.bottleneck = nn.BatchNorm1d(dim)
        self.classifier = nn.Linear(dim, num_classes, bias=False)
        self.dropout = nn.Dropout(cfg.MODEL.DROPOUT_RATE)
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.patch_compensation = PatchCompensation(
            dim, cfg.MODEL.PATCH_HIDDEN_DIM, cfg.MODEL.PATCH_HEADS,
            cfg.MODEL.PATCH_LAYERS, cfg.MODEL.PATCH_USE_NULL, cfg.MODEL.PATCH_ZERO_OUTPUT)
        allow_baseline = Path(cfg.MODEL.PRETRAIN_PATH).resolve() == Path(self.baseline_checkpoint).resolve()
        self.load_param(cfg.MODEL.PRETRAIN_PATH, allow_baseline=allow_baseline)
        # requires_grad alone does not freeze BN statistics or stochastic depth.
        for module in (self.base, self.mldc, self.bottleneck):
            module.requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        self.mldc.eval()
        self.bottleneck.eval()
        return self

    def forward(self, x, label=None, cam_label=None, view_label=None,
                return_components=False, return_patch_diagnostics=False):
        if return_patch_diagnostics and self.training:
            raise ValueError('Patch diagnostics are an evaluation-only return format')
        # Frozen parameters need no graph for normal inputs. Do not force no_grad:
        # input-gradient attribution must remain possible on this same path.
        layers, patches = self.base(x, cam_label=cam_label, view_label=view_label,
                                    return_patches=True, patch_layers=self.patch_compensation.layers)
        anchor = self.mldc(layers)
        result = self.patch_compensation(anchor, patches, return_attention=return_patch_diagnostics)
        if return_patch_diagnostics:
            delta, patch_stats = result
        else:
            delta = result
        feature = anchor + delta
        neck = self.bottleneck(feature)
        if self.training:
            return self.classifier(self.dropout(neck)), feature
        output = neck if self.neck_feat == 'after' else feature
        if return_patch_diagnostics:
            return {'output': output, 'raw': feature, 'anchor': anchor, 'delta': delta,
                    'cls_features': layers, **patch_stats}
        return (output, feature, layers) if return_components else output

    def readout_weights(self):
        return self.mldc.normalized_weights()

    def checkpoint_metadata(self):
        return {'experiment': EXPERIMENT, 'stage': 'frozen_anchor',
                'patch_architecture': self.patch_compensation.architecture(),
                'baseline_checkpoint': self.baseline_checkpoint,
                'baseline_epoch': self.baseline_epoch}

    def load_param(self, path, allow_baseline=False):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if tuple(checkpoint.get('mldc_layers', ())) != self.mldc.selected_layers:
            raise ValueError('Checkpoint must use exactly CLS layers 9, 10, 11, 12')
        if checkpoint.get('readout') != 'mldc' or checkpoint.get('temperature') != self.mldc.temperature:
            raise ValueError('Checkpoint CLS readout or temperature differs from the baseline')
        state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
        if checkpoint.get('experiment') == EXPERIMENT:
            if (checkpoint.get('patch_architecture') != self.patch_compensation.architecture()
                    or checkpoint.get('stage') != 'frozen_anchor'
                    or checkpoint.get('train_feature') != self.train_feature):
                raise ValueError('Patch checkpoint architecture, stage, or supervision differs')
            self.load_state_dict(state, strict=True)
            self.checkpoint_epoch = int(checkpoint.get('epoch', 0))
            self.baseline_epoch = int(checkpoint.get('baseline_epoch', 120))
        else:
            if (not allow_baseline or checkpoint.get('train_feature') != 'mldc'
                    or any(k.startswith('patch_compensation.') for k in state)):
                raise ValueError('Only the explicitly configured baseline may initialize a new experiment')
            expected_anchor = {k for k in self.state_dict() if not k.startswith('patch_compensation.')}
            if set(state) != expected_anchor:
                raise ValueError('Baseline checkpoint contains missing or unexpected anchor keys')
            incompatible = self.load_state_dict(state, strict=False)
            expected_missing = {k for k in self.state_dict() if k.startswith('patch_compensation.')}
            if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
                raise RuntimeError('Baseline loading omitted something other than new patch parameters')
            self.baseline_epoch = int(checkpoint.get('epoch', 0))
            self.checkpoint_epoch = 0


def make_model(cfg, num_class, camera_num, view_num):
    model = MLDCReID(cfg, num_class, camera_num, view_num)
    print('{}: stage=frozen_anchor, CLS={}, patch={}, hidden={}, heads={}, null={}, descriptor=768'.format(
        EXPERIMENT, model.mldc.selected_layers, model.patch_compensation.layers,
        model.patch_compensation.hidden_dim, model.patch_compensation.heads,
        model.patch_compensation.use_null))
    return model
