import torch
from torch import nn
from .lpam import FeaturePyramid

VARIANT = 'mldc_deep4_fixed_equal_lpam_fft_v4_20261017'
BRANCHES = ('global', 'f3', 'f5', 'f6')


def add_config(cfg):
    cfg.MODEL.LPAM_ENABLED = True
    cfg.MODEL.GLOBAL_LAYERS = [9, 10, 11, 12]
    cfg.MODEL.GLOBAL_FUSION = 'fixed_equal'
    cfg.MODEL.LPAM_LAYERS = [3, 6, 9, 12]
    cfg.MODEL.LPAM_FFT = True
    cfg.SOLVER.MLDC_HEAD_LR_FACTOR = 5.0
    return cfg


def fixed_equal_global(layers):
    if layers.ndim != 3 or layers.shape[0] != 12:
        raise ValueError('Backbone must return 12 raw post-block CLS features')
    return layers[8:12].mean(0)


class LPAMFFTReID(nn.Module):
    def __init__(self, cfg, num_classes, cameras, views, initialize=True):
        super().__init__()
        from .backbones.vit_pytorch_mldc import vit_base_patch16_224_TransReID
        if list(cfg.MODEL.GLOBAL_LAYERS) != [9,10,11,12] or cfg.MODEL.GLOBAL_FUSION != 'fixed_equal':
            raise ValueError('Global fusion must stay exactly equal throughout training')
        if list(cfg.MODEL.LPAM_LAYERS) != [3,6,9,12] or not cfg.MODEL.LPAM_FFT:
            raise ValueError('Use author FFT LPAM and CLS3/6/9/12')
        if (cfg.MODEL.PRETRAIN_CHOICE != 'imagenet' or cfg.MODEL.PRETRAIN_PATH != 'pretrain_model/vit_base_ics_cfs_lup.pth'
                or cfg.MODEL.JPM or cfg.MODEL.USE_DCE or cfg.MODEL.QNORM_PRIOR or cfg.MODEL.QNORM_GUIDE
                or cfg.MODEL.GEM_POOLING or cfg.MODEL.DIST_TRAIN or cfg.MODEL.REDUCE_FEAT_DIM
                or cfg.MODEL.TRANSFORMER_TYPE != 'vit_base_patch16_224_TransReID'
                or cfg.MODEL.ID_LOSS_TYPE != 'softmax' or cfg.TEST.NECK_FEAT != 'after'):
            raise ValueError('Use the matched SSL backbone, softmax head and BN-after retrieval')
        # Preserve original MLDC random-number consumption before the backbone;
        # fusion itself has no parameters and never learns unequal weights.
        torch.randn(12)
        self.base = vit_base_patch16_224_TransReID(
            img_size=cfg.INPUT.SIZE_TRAIN, stride_size=cfg.MODEL.STRIDE_SIZE,
            camera=cameras if cfg.MODEL.SIE_CAMERA else 0,
            view=views if cfg.MODEL.SIE_VIEW else 0, sie_xishu=cfg.MODEL.SIE_COE,
            stem_conv=cfg.MODEL.STEM_CONV, drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate=cfg.MODEL.DROP_OUT, attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if initialize:
            self.base.load_param(cfg.MODEL.PRETRAIN_PATH, hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
        # These modules are not used by the raw per-block CLS forward path.
        for module in (self.base.norm, self.base.fc, self.base.gem):
            module.requires_grad_(False)
        dim = self.base.in_planes
        self.bottleneck = nn.BatchNorm1d(dim)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=.001)
        self.dropout = nn.Dropout(cfg.MODEL.DROPOUT_RATE)
        self.enabled = bool(cfg.MODEL.LPAM_ENABLED)
        self.lpam = FeaturePyramid(dim) if self.enabled else None
        self.register_buffer('global_weights', torch.tensor([0.]*8+[.25]*4))
        self.checkpoint_epoch = 0

    def from_layers(self, layers):
        global_raw = fixed_equal_global(layers)
        local = self.lpam(*(layers[i-1].unsqueeze(-1) for i in (3,6,9,12))) if self.enabled else ()
        raw = (global_raw,)+tuple(x.squeeze(-1) for x in local)
        # Author shared neck and classifier are deliberately retained.
        neck = tuple(self.bottleneck(x) for x in raw)
        result = {'raw_global':global_raw, 'layers':layers,
                  'retrieval':torch.cat(neck, dim=-1), 'global':neck[0]}
        if self.enabled:
            result['local'] = torch.cat(neck[1:],dim=-1)
            result.update(dict(zip(BRANCHES[1:],neck[1:])))
        if self.training:
            result['scores'] = tuple(self.classifier(self.dropout(x)) for x in neck)
        return result

    def forward(self, images, label=None, cam_label=None, view_label=None, diagnostics=False):
        layers = self.base(images, cam_label=cam_label, view_label=view_label)
        result = self.from_layers(layers)
        return result if self.training or diagnostics else result['retrieval']

    def readout_weights(self):
        return self.global_weights

    def architecture(self):
        return dict(variant=VARIANT, global_layers=[9,10,11,12], global_fusion='fixed_equal',
                    lpam_enabled=self.enabled, lpam_layers=[3,6,9,12], lpam_input='raw_cls',
                    tokens_per_input=1, fft=True, conv_attention_residual=False,
                    shared_pyramid_block=True, shared_bn_classifier=True,
                    descriptor_dim=3072 if self.enabled else 768,
                    retrieval='direct_concat_bn_after_then_evaluator_l2',
                    joint_from_epoch=1, spatial_partition=False)


def objectives(data, labels, cfg):
    import torch.nn.functional as F
    from loss.triplet_loss import euclidean_dist, hard_example_mining
    with torch.autocast(device_type=labels.device.type, enabled=False):
        ids = torch.stack([F.cross_entropy(s.float(),labels) for s in data['scores']])
        positive, negative = hard_example_mining(euclidean_dist(data['raw_global'].float(),data['raw_global'].float()), labels)
        triplet = F.softplus(positive-negative).mean() if cfg.MODEL.NO_MARGIN else F.relu(positive-negative+cfg.SOLVER.MARGIN).mean()
        loss = cfg.MODEL.ID_LOSS_WEIGHT*ids.mean()+cfg.MODEL.TRIPLET_LOSS_WEIGHT*triplet
        if not torch.isfinite(loss).item():
            raise FloatingPointError('Non-finite V4 FFT objective')
        return loss, ids.detach(), triplet.detach()
