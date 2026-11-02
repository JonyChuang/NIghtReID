"""Single-branch Night600 MLDC experiment and matched last-CLS control."""
import torch
from torch import nn
from .backbones.vit_pytorch_mldc import vit_base_patch16_224_TransReID
from .mldc import MLDC


def add_mldc_config(cfg):
    cfg.MODEL.MLDC_READOUT = 'mldc'
    cfg.MODEL.MLDC_TEMPERATURE = 0.9
    cfg.MODEL.MLDC_INIT = 'random'
    cfg.MODEL.MLDC_DEEP_INIT_MASS = 0.75
    cfg.SOLVER.MLDC_HEAD_LR_FACTOR = 5.0
    cfg.SOLVER.MLDC_WEIGHT_LR_FACTOR = 1.0
    cfg.TEST.MLDC_VISUALIZE = True
    cfg.TEST.MLDC_VIS_SAMPLES = 6
    cfg.TEST.MLDC_VIS_PERIOD = 10
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
        self.checkpoint_epoch = 0
        # Instantiate in both modes so their backbone and classifier consume
        # exactly the same RNG sequence under the same seed.
        self.mldc = MLDC(12, cfg.MODEL.MLDC_TEMPERATURE,
                         init=getattr(cfg.MODEL, 'MLDC_INIT', 'random'),
                         deep_mass=getattr(cfg.MODEL, 'MLDC_DEEP_INIT_MASS', 0.75))
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

    def forward(self, x, label=None, cam_label=None, view_label=None, return_components=False):
        features = self.base(x, cam_label=cam_label, view_label=view_label)
        feature = self.mldc(features) if self.readout == 'mldc' else features[-1]
        neck = self.bottleneck(feature)
        if self.training:
            return self.classifier(self.dropout(neck)), feature
        output = neck if self.neck_feat == 'after' else feature
        return (output, feature, features) if return_components else output

    def readout_weights(self):
        if self.readout == 'mldc':
            return self.mldc.normalized_weights()
        weights = torch.zeros_like(self.mldc.layer_weights)
        weights[-1] = 1
        return weights

    def load_param(self, path):
        checkpoint = torch.load(path, map_location='cpu')
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
    print('A3PFN MLDC experiment: readout={}, init={}, 12 post-block raw CLS, temperature={}, descriptor=768'.format(
        model.readout, model.mldc.init, model.mldc.temperature))
    return model
