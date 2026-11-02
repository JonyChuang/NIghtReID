"""MLDC token fusion followed by the LocAt readout, with controlled ablations."""
import torch
from torch import nn
from .make_model_mldc import MLDCReID, add_mldc_config as add_baseline_config
from .make_model_mldc import configure_head_learning_rates
from .locat import GaussianAttention, PRR


def add_mldc_config(cfg):
    add_baseline_config(cfg)
    cfg.MODEL.LOCAT_GAUG = True
    cfg.MODEL.LOCAT_PRR = True
    cfg.MODEL.LOCAT_READOUT_NORM = True
    return cfg


class LocAtReID(MLDCReID):
    def __init__(self, cfg, num_classes, camera_num, view_num):
        # Load an experiment checkpoint only after the new modules exist.
        construction = cfg.clone()
        construction.defrost()
        restoring = cfg.MODEL.PRETRAIN_CHOICE == 'self'
        if restoring:
            construction.MODEL.PRETRAIN_CHOICE = 'imagenet'
            construction.MODEL.PRETRAIN_PATH = ''
        construction.freeze()
        # The baseline API does not support random initialization; for test
        # construction and checkpoint restoration skip its SSL loader explicitly.
        self._skip_ssl_load = restoring or not cfg.MODEL.PRETRAIN_PATH
        if self._skip_ssl_load:
            construction = construction.clone()
            construction.defrost()
            construction.MODEL.PRETRAIN_CHOICE = 'random'
            construction.freeze()
        super().__init__(construction, num_classes, camera_num, view_num)
        self.locat_spec = {'gaug': bool(cfg.MODEL.LOCAT_GAUG),
                           'prr': bool(cfg.MODEL.LOCAT_PRR),
                           'readout_norm': bool(cfg.MODEL.LOCAT_READOUT_NORM)}
        if self.readout != 'mldc':
            raise ValueError('LocAt tests preserve the MLDC readout')
        if self.locat_spec['prr'] and not self.locat_spec['readout_norm']:
            raise ValueError('PRR must use the official normalized input convention')
        grid = (self.base.patch_embed.num_y, self.base.patch_embed.num_x)
        # All configurations share baseline MLDC/backbone/classifier RNG state.
        # GAug initialization must not change the subsequent data/dropout RNG.
        if self.locat_spec['gaug']:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(cfg.SOLVER.SEED) + 1703)
                for block in self.base.blocks:
                    block.attn = GaussianAttention(block.attn, self.base.in_planes, grid)
                    block.attn.gaug.apply(self.base._init_weights)
        self.prr = PRR(self.base.in_planes, self.base.blocks[0].attn.num_heads)
        self.base.norm.requires_grad_(self.locat_spec['readout_norm'])
        if restoring:
            self.load_param(cfg.MODEL.PRETRAIN_PATH)

    def forward(self, x, label=None, cam_label=None, view_label=None,
                return_components=False, return_details=False):
        weights = self.mldc.normalized_weights()
        layers, tokens = self.base(x, cam_label=cam_label, view_label=view_label, token_weights=weights)
        raw_mldc = self.mldc(layers)
        normalized = self.base.norm(tokens) if self.locat_spec['readout_norm'] else tokens
        feature = self.prr(normalized) if self.locat_spec['prr'] else (
            normalized[:, 0] if self.locat_spec['readout_norm'] else raw_mldc)
        neck = self.bottleneck(feature)
        if self.training and not return_details:
            return self.classifier(self.dropout(neck)), feature
        output = neck if self.neck_feat == 'after' else feature
        if return_details:
            return {'retrieval': output, 'raw_readout': feature, 'raw_mldc': raw_mldc,
                    'raw_last_cls': layers[-1], 'raw_uniform_cls': layers.mean(0),
                    'prr_input_cls': normalized[:, 0], 'layers': layers,
                    'fused_tokens': tokens}
        return (output, feature, layers) if return_components else output

    def load_param(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        if checkpoint.get('locat_spec') != self.locat_spec:
            raise ValueError('Checkpoint LocAt modules do not match this configuration')
        if checkpoint.get('temperature', self.mldc.temperature) != self.mldc.temperature:
            raise ValueError('Checkpoint MLDC temperature mismatch')
        state = checkpoint.get('state_dict', checkpoint)
        state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
        self.load_state_dict(state, strict=True)
        self.checkpoint_epoch = int(checkpoint.get('epoch', 0))


def make_model(cfg, num_class, camera_num, view_num):
    model = LocAtReID(cfg, num_class, camera_num, view_num)
    print('MLDC + LocAt:', model.locat_spec, '| MLDC token fusion -> norm -> PRR -> 768D BN neck')
    return model
