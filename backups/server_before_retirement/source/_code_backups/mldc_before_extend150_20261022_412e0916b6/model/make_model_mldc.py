"""Single-branch Night600 MLDC experiment and matched last-CLS control."""
import torch
from torch import nn
from .backbones.vit_pytorch_mldc import vit_base_patch16_224_TransReID
from .mldc import MLDC
from .reference_space import shared_bn_references


def add_mldc_config(cfg):
    cfg.MODEL.MLDC_READOUT = 'mldc'
    cfg.MODEL.MLDC_TEMPERATURE = 0.9
    cfg.MODEL.MLDC_LAYERS = [9, 10, 11, 12]
    cfg.MODEL.MLDC_INIT = 'uniform'
    cfg.MODEL.MLDC_TRAIN_FEATURE = 'mldc'
    cfg.MODEL.MLDC_DEEP_INIT_MASS = 0.75
    cfg.SOLVER.MLDC_HEAD_LR_FACTOR = 5.0
    cfg.SOLVER.MLDC_WEIGHT_LR_FACTOR = 1.0
    cfg.TEST.MLDC_VISUALIZE = True
    cfg.TEST.MLDC_VIS_SAMPLES = 6
    cfg.TEST.MLDC_VIS_PERIOD = 10
    cfg.TEST.MLDC_RETRIEVAL = 'mldc'
    cfg.SOLVER.PRESERVE_WEIGHT = 0.1
    cfg.SOLVER.PRESERVE_MIN_MARGIN = 0.05
    cfg.SOLVER.PRESERVE_MAX_MARGIN = 0.2
    cfg.SOLVER.PRESERVE_MOMENTUM = 0.2
    cfg.SOLVER.PRESERVE_HARD_NEGATIVES = 5
    cfg.SOLVER.PRESERVE_REFERENCE_SPACE = 'shared_bn'
    cfg.SOLVER.PRESERVE_TARGET_MODE = 'ratio'
    cfg.SOLVER.PRESERVE_RETAIN_RATIO = 0.8
    cfg.SOLVER.AUXILIARY_LOSS = 'preservation'
    cfg.SOLVER.CNL_WEIGHT = 0.1
    cfg.SOLVER.CNL_WARMUP_EPOCHS = 0
    cfg.SOLVER.CNL_GAMMA = 0.9
    cfg.SOLVER.CNL_TEMPERATURE = 0.05
    cfg.SOLVER.CNL_MOMENTUM = 0.2
    cfg.SOLVER.CNL_CROSS_CAMERA = True
    cfg.SOLVER.CNL_SELECTION = 'union'
    cfg.SOLVER.CNL_MIN_SUPPORT = 3
    cfg.SOLVER.CNL_UNCERTAIN_GAMMA = 0.8
    return cfg


def configure_head_learning_rates(cfg, model, optimizer):
    heads = {id(p) for name, p in model.named_parameters()
             if p.requires_grad and name.startswith(('bottleneck.', 'classifier.'))}
    for group in optimizer.param_groups:
        ids = {id(p) for p in group['params']}
        if ids & heads:
            if not ids <= heads:
                raise RuntimeError('Optimizer mixes head and backbone parameters')
            group['lr'] *= cfg.SOLVER.MLDC_HEAD_LR_FACTOR


class MLDCReID(nn.Module):
    def __init__(self, cfg, num_classes, camera_num, view_num):
        super().__init__()
        if cfg.MODEL.NAME != 'transformer' or cfg.MODEL.TRANSFORMER_TYPE != 'vit_base_patch16_224_TransReID':
            raise ValueError('This experiment uses the Night600 ViT-Base backbone')
        if cfg.MODEL.JPM or cfg.MODEL.GEM_POOLING or cfg.MODEL.USE_DCE or cfg.MODEL.QNORM_PRIOR or cfg.MODEL.QNORM_GUIDE:
            raise ValueError('MLDC experiment requires the plain backbone')
        if cfg.MODEL.DIST_TRAIN or cfg.MODEL.REDUCE_FEAT_DIM or cfg.MODEL.ID_LOSS_TYPE != 'softmax':
            raise ValueError('MLDC experiment uses single-GPU softmax training with full features')
        if cfg.MODEL.PRETRAIN_CHOICE not in ('imagenet', 'self'):
            raise ValueError('Use imagenet for SSL initialization or self for an experiment checkpoint')
        if cfg.MODEL.MLDC_READOUT not in ('mldc', 'last_cls') or cfg.TEST.NECK_FEAT not in ('before', 'after'):
            raise ValueError('Invalid MLDC readout or neck feature setting')
        self.readout = cfg.MODEL.MLDC_READOUT
        self.preservation_reference_space = cfg.SOLVER.PRESERVE_REFERENCE_SPACE
        if self.preservation_reference_space not in ('raw_cls', 'shared_bn'):
            raise ValueError('Preservation reference space must be raw_cls or shared_bn')
        self.train_feature = getattr(cfg.MODEL, 'MLDC_TRAIN_FEATURE', 'mldc')
        if self.train_feature not in ('mldc', 'residual_mean'):
            raise ValueError('MLDC training feature must be mldc or residual_mean')
        self.retrieval = getattr(cfg.TEST, 'MLDC_RETRIEVAL', 'mldc')
        if self.retrieval not in ('mldc', 'raw_residual', 'supervised_residual'):
            raise ValueError('Unknown MLDC retrieval mode')
        if self.retrieval in ('raw_residual', 'supervised_residual') and (self.readout != 'mldc' or cfg.TEST.NECK_FEAT != 'before' or cfg.TEST.FEAT_NORM != 'yes'):
            raise ValueError('Residual retrieval requires MLDC, before-BN features, and final L2 normalization')
        if (self.train_feature == 'residual_mean') != (self.retrieval == 'supervised_residual'):
            raise ValueError('Residual-mean supervision requires matching supervised-residual retrieval')
        self.checkpoint_epoch = 0
        # Instantiate in both modes so their backbone and classifier consume
        # exactly the same RNG sequence under the same seed.
        self.mldc = MLDC(12, cfg.MODEL.MLDC_TEMPERATURE,
                         init=getattr(cfg.MODEL, 'MLDC_INIT', 'uniform'),
                         deep_mass=getattr(cfg.MODEL, 'MLDC_DEEP_INIT_MASS', 0.75),
                         selected_layers=getattr(cfg.MODEL, 'MLDC_LAYERS', list(range(1, 13))))
        self.mldc.layer_weights.requires_grad_(self.readout == 'mldc')
        self.base = vit_base_patch16_224_TransReID(
            img_size=cfg.INPUT.SIZE_TRAIN, stride_size=cfg.MODEL.STRIDE_SIZE,
            camera=camera_num if cfg.MODEL.SIE_CAMERA else 0,
            view=view_num if cfg.MODEL.SIE_VIEW else 0,
            sie_xishu=cfg.MODEL.SIE_COE, stem_conv=cfg.MODEL.STEM_CONV,
            drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            self.base.load_param(cfg.MODEL.PRETRAIN_PATH, hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
        # Retain pretrained state keys but exclude inactive layers from SGD.
        for module in (self.base.norm, self.base.fc, self.base.gem):
            module.requires_grad_(False)
        dim = self.base.in_planes
        self.bottleneck = nn.BatchNorm1d(dim)
        nn.init.ones_(self.bottleneck.weight)
        nn.init.zeros_(self.bottleneck.bias)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        self.dropout = nn.Dropout(cfg.MODEL.DROPOUT_RATE)
        self.neck_feat = cfg.TEST.NECK_FEAT
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            self.load_param(cfg.MODEL.PRETRAIN_PATH)

    def forward(self, x, label=None, cam_label=None, view_label=None, return_components=False,
                return_preservation_features=False, return_cnl_features=False):
        features = self.base(x, cam_label=cam_label, view_label=view_label)
        feature = self.mldc(features) if self.readout == 'mldc' else features[-1]
        supervised = (feature + features[-1]) * 0.5 if self.train_feature == 'residual_mean' else feature
        neck = self.bottleneck(supervised)
        if self.training:
            scores = self.classifier(self.dropout(neck))
            if return_cnl_features:
                layers = features.index_select(0, self.mldc.selected_indices).permute(1, 0, 2)
                # Layers mine neighbors; CNL gradients flow through the actual
                # fused BN-after retrieval descriptor returned below.
                layers = shared_bn_references(layers, supervised, self.bottleneck)
                return scores, supervised, torch.nn.functional.normalize(neck.float(), dim=1), layers
            if return_preservation_features:
                layers = features.index_select(0, self.mldc.selected_indices).permute(1, 0, 2)
                if self.preservation_reference_space == 'shared_bn':
                    layers = shared_bn_references(layers, supervised, self.bottleneck)
                return scores, supervised, torch.nn.functional.normalize(neck.float(), dim=1), layers.detach()
            return scores, supervised
        if self.retrieval == 'raw_residual':
            # Fixed coefficient 1; final L2 is applied by the retrieval evaluator.
            output = feature + features[-1]
        elif self.retrieval == 'supervised_residual':
            output = supervised
        else:
            output = neck if self.neck_feat == 'after' else feature
        return (output, feature, features) if return_components else output

    def readout_weights(self):
        if self.readout == 'mldc':
            return self.mldc.normalized_weights()
        weights = self.mldc.layer_weights.new_zeros(self.mldc.depth)
        weights[-1] = 1
        return weights

    def load_param(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        if tuple(checkpoint.get('mldc_layers', range(1, 13))) != self.mldc.selected_layers:
            raise ValueError('Checkpoint MLDC layers do not match the selected layers')
        if checkpoint.get('train_feature', 'mldc') != self.train_feature:
            raise ValueError('Checkpoint training feature does not match this experiment')
        if 'readout' in checkpoint and checkpoint['readout'] != self.readout:
            raise ValueError('Checkpoint readout does not match the selected experiment')
        if 'temperature' in checkpoint and checkpoint['temperature'] != self.mldc.temperature:
            raise ValueError('Checkpoint MLDC temperature does not match the configuration')
        state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
        self.load_state_dict(state, strict=True)
        self.checkpoint_epoch = int(checkpoint.get('epoch', 0))


def make_model(cfg, num_class, camera_num, view_num):
    model = MLDCReID(cfg, num_class, camera_num, view_num)
    print('A3PFN MLDC experiment: readout={}, init={}, train_feature={}, retrieval={}, selected_layers={}, temperature={}, descriptor=768'.format(
        model.readout, model.mldc.init, model.train_feature, model.retrieval,
        model.mldc.selected_layers, model.mldc.temperature))
    return model
