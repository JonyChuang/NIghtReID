import torch
from torch import nn
from .featenhancer import FeatEnHancer
from .patch_enhancer import PatchCompensation

LEGACY_VARIANT='featenhancer_mldc_deep4_fixed_equal_20261017'
VARIANT='featenhancer_patch_mldc_deep4_fixed_equal_scheme_b_20261017'


def add_config(cfg):
    cfg.MODEL.FEATENHANCER_ENABLED=True
    cfg.MODEL.ENHANCEMENT_MODE='patch_feature'
    cfg.MODEL.PATCH_LAYERS=[4,6,8]
    cfg.MODEL.PATCH_DIM=192
    cfg.MODEL.PATCH_HIDDEN=64
    cfg.MODEL.PATCH_GAIN=.1
    cfg.MODEL.GLOBAL_LAYERS=[9,10,11,12]
    cfg.MODEL.GLOBAL_FUSION='fixed_equal'
    cfg.SOLVER.MLDC_HEAD_LR_FACTOR=5.0
    cfg.TEST.PATCH_VISUALIZE=True
    cfg.TEST.PATCH_VIS_INDICES=[810,1544,11,2059,160,24]
    return cfg


def config_from_dump(base_cfg,text):
    cfg=add_config(base_cfg.clone())
    stored=type(cfg).load_cfg(text)
    if 'ENHANCEMENT_MODE' not in stored.MODEL:
        cfg.MODEL.ENHANCEMENT_MODE='input'
    # Preserve the device string across YACS's second literal decoding pass.
    stored.MODEL.DEVICE_ID=repr(str(stored.MODEL.DEVICE_ID))
    cfg.merge_from_other_cfg(stored)
    return cfg


class FixedEqualMLDC(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weights',torch.tensor([0.]*8+[.25]*4))

    def forward(self,layers):
        if layers.ndim!=3 or layers.shape[0]!=12:
            raise ValueError('Expected all 12 raw post-block CLS features')
        return layers[8:12].mean(0)

    def normalized_weights(self):
        return self.weights


def author_initialize_enhancer(module):
    # Same Conv2d weight initialization as the author's detector wrapper.
    # Biases retain the defaults of the original nn.Conv2d constructors.
    if isinstance(module,nn.Conv2d):
        nn.init.normal_(module.weight,mean=0.,std=.02)


class FeatEnHancerMLDC(nn.Module):
    def __init__(self,cfg,num_classes,cameras,views,initialize=True):
        super().__init__()
        from .backbones.vit_pytorch_mldc import vit_base_patch16_224_TransReID
        if list(cfg.MODEL.GLOBAL_LAYERS)!=[9,10,11,12] or cfg.MODEL.GLOBAL_FUSION!='fixed_equal':
            raise ValueError('Use fixed arithmetic mean of CLS9-12')
        if (cfg.MODEL.PRETRAIN_CHOICE!='imagenet' or cfg.MODEL.PRETRAIN_PATH!='pretrain_model/vit_base_ics_cfs_lup.pth'
                or cfg.MODEL.TRANSFORMER_TYPE!='vit_base_patch16_224_TransReID'
                or cfg.MODEL.ID_LOSS_TYPE!='softmax' or cfg.MODEL.JPM or cfg.MODEL.USE_DCE
                or cfg.MODEL.QNORM_PRIOR or cfg.MODEL.QNORM_GUIDE or cfg.MODEL.GEM_POOLING
                or cfg.MODEL.DIST_TRAIN or cfg.MODEL.REDUCE_FEAT_DIM or cfg.TEST.NECK_FEAT!='after'):
            raise ValueError('Use the matched plain Night600 SSL backbone and BN-after retrieval')
        # Match baseline RNG consumption; no trainable layer weights exist.
        torch.randn(12)
        self.base=vit_base_patch16_224_TransReID(
            img_size=cfg.INPUT.SIZE_TRAIN,stride_size=cfg.MODEL.STRIDE_SIZE,
            camera=cameras if cfg.MODEL.SIE_CAMERA else 0,
            view=views if cfg.MODEL.SIE_VIEW else 0,sie_xishu=cfg.MODEL.SIE_COE,
            stem_conv=cfg.MODEL.STEM_CONV,drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate=cfg.MODEL.DROP_OUT,attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if initialize:
            self.base.load_param(cfg.MODEL.PRETRAIN_PATH,hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
        # Exclude only modules unused by this backbone's raw CLS output path.
        for module in (self.base.norm,self.base.fc,self.base.gem):
            module.requires_grad_(False)
        self.mldc=FixedEqualMLDC()
        dim=self.base.in_planes
        self.bottleneck=nn.BatchNorm1d(dim)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier=nn.Linear(dim,num_classes,bias=False)
        nn.init.normal_(self.classifier.weight,std=.001)
        self.dropout=nn.Dropout(cfg.MODEL.DROPOUT_RATE)
        self.enabled=bool(cfg.MODEL.FEATENHANCER_ENABLED)
        self.mode=cfg.MODEL.ENHANCEMENT_MODE
        if self.mode not in ('input','patch_feature'):
            raise ValueError('Unknown enhancement mode')
        # Build after backbone/head initialization to keep the matched control
        # backbone and classifier initialization identical under the same seed.
        self.feat_enhancer=FeatEnHancer(3) if self.enabled and self.mode=='input' else nn.Identity()
        self.patch_layers=tuple(cfg.MODEL.PATCH_LAYERS)
        if self.enabled and self.mode=='input':
            self.feat_enhancer.apply(author_initialize_enhancer)
        if self.enabled and self.mode=='patch_feature':
            self.patch_compensation=PatchCompensation(dim,cfg.MODEL.PATCH_DIM,cfg.MODEL.PATCH_HIDDEN,
                                                     self.patch_layers,cfg.MODEL.PATCH_GAIN)
        self.checkpoint_epoch=0

    def forward(self,images,label=None,cam_label=None,view_label=None,diagnostics=False):
        # Inputs already underwent the original ReID normalization exactly once.
        enhanced=self.feat_enhancer(images)
        if enhanced.shape!=images.shape:
            raise RuntimeError('FeatEnHancer must preserve [B,3,H,W]')
        patch_data={}
        if self.enabled and self.mode=='patch_feature':
            layers,patches=self.base(images,cam_label=cam_label,view_label=view_label,patch_layers=self.patch_layers)
            global_feature=self.mldc(layers)
            grid=(self.base.patch_embed.num_y,self.base.patch_embed.num_x)
            patch_data=self.patch_compensation(patches,global_feature,grid,diagnostics=diagnostics)
            raw=patch_data['compensated']
        else:
            layers=self.base(enhanced,cam_label=cam_label,view_label=view_label)
            global_feature=self.mldc(layers)
            raw=global_feature
        neck=self.bottleneck(raw)
        if self.training or diagnostics:
            result={'retrieval':neck,'raw_readout':raw,'global_feature':global_feature,'layers':layers,'enhanced':enhanced,**patch_data}
            if self.training:
                result['scores']=self.classifier(self.dropout(neck))
            return result
        return neck

    def readout_weights(self):
        return self.mldc.normalized_weights()

    def architecture(self):
        if self.mode=='patch_feature':
            return dict(variant=VARIANT,enhancer_enabled=self.enabled,enhancer='adapted patch-space FEN/SAFA',
                        patch_layers=list(self.patch_layers),patch_dim=self.patch_compensation.output.in_features if self.enabled else None,
                        patch_gain=self.patch_compensation.gain if self.enabled else 0.,
                        patch_readout='CLS-conditioned cosine attention, spatially centered values, equal depth mean',
                        global_layers=[9,10,11,12],global_fusion='fixed_equal',global_weights=[.25]*4,
                        descriptor_dim=768,spatial_partition=False,joint_from_epoch=1,retrieval='BN-after compensated global; evaluator L2')
        return dict(variant=LEGACY_VARIANT,enhancer_enabled=self.enabled,
                    enhancer='author FeatEnHancer at normalized input, before backbone',
                    global_layers=[9,10,11,12],global_fusion='fixed_equal',
                    global_weights=[.25,.25,.25,.25],descriptor_dim=768,
                    spatial_partition=False,joint_from_epoch=1,
                    retrieval='BN-after global; evaluator L2')


def objectives(data,labels,cfg):
    import torch.nn.functional as F
    from loss.triplet_loss import euclidean_dist,hard_example_mining
    with torch.autocast(device_type=labels.device.type,enabled=False):
        identity=F.cross_entropy(data['scores'].float(),labels)
        feature=data['raw_readout'].float()
        positive,negative=hard_example_mining(euclidean_dist(feature,feature),labels)
        triplet=F.softplus(positive-negative).mean() if cfg.MODEL.NO_MARGIN else F.relu(positive-negative+cfg.SOLVER.MARGIN).mean()
        loss=cfg.MODEL.ID_LOSS_WEIGHT*identity+cfg.MODEL.TRIPLET_LOSS_WEIGHT*triplet
        if not torch.isfinite(loss).item():
            raise FloatingPointError('Non-finite FeatEnHancer MLDC identity objective')
        return loss,identity.detach(),triplet.detach()
