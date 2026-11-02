"""MLDC+CNL with per-block ViTMatte convolution Necks and the DCM branch."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .base_model import MLDCReID as Baseline, add_mldc_config as base_config
from .detail_branch import ViTMatteDetails
from .conv_neck import attach_conv_necks

VARIANT='vitmatte_block_neck_detail_mldc_cnl_20261025'


def add_mldc_config(cfg):
    cfg=base_config(cfg)
    cfg.MODEL.DETAIL_ENABLED=True
    cfg.MODEL.DETAIL_DIM=256
    cfg.MODEL.DETAIL_ALPHA=0.1
    cfg.MODEL.DETAIL_POOLING='guided'
    cfg.MODEL.CONV_NECK_ENABLED=True
    cfg.MODEL.CONV_NECK_LAYERS=list(range(1,13))
    cfg.MODEL.CONV_NECK_PLACEMENT='post_mlp'
    cfg.SOLVER.DETAIL_LOSS_WEIGHT=1.0
    cfg.SOLVER.DETAIL_LR_FACTOR=1.0
    cfg.TEST.MLDC_VISUALIZE=False
    return cfg


def configure_head_learning_rates(cfg,model,optimizer):
    head_prefix=('bottleneck.','classifier.','detail_bottleneck.','fused_bottleneck.','fused_classifier.')
    heads={id(p) for n,p in model.named_parameters() if p.requires_grad and n.startswith(head_prefix)}
    detail={id(p) for n,p in model.named_parameters() if p.requires_grad and n.startswith('detail_branch.')}
    for group in optimizer.param_groups:
        ids={id(p) for p in group['params']}
        factor=1.
        if ids&heads:
            assert ids<=heads
            factor=cfg.SOLVER.MLDC_HEAD_LR_FACTOR
        elif ids&detail:
            assert ids<=detail
            factor=cfg.SOLVER.DETAIL_LR_FACTOR
        group['lr']*=factor


def fuse_units(global_unit,detail_unit,alpha):
    return torch.cat((math.sqrt(1-alpha)*global_unit,math.sqrt(alpha)*detail_unit),dim=1)


class MLDCReID(Baseline):
    def __init__(self,cfg,num_classes,camera_num,view_num,initialize=True):
        if cfg.SOLVER.SEED!=1234:
            raise ValueError('This experiment fixes seed 1234')
        if (cfg.MODEL.MLDC_READOUT!='mldc' or list(cfg.MODEL.MLDC_LAYERS)!=[9,10,11,12]
            or cfg.MODEL.MLDC_TRAIN_FEATURE!='mldc' or cfg.TEST.MLDC_RETRIEVAL!='mldc'
            or cfg.TEST.NECK_FEAT!='after' or cfg.TEST.FEAT_NORM!='yes'
            or list(cfg.MODEL.STRIDE_SIZE)!=[16,16]):
            raise ValueError('Use the original learned deep-four BN-after, stride-16 global path')
        super().__init__(cfg,num_classes,camera_num,view_num,initialize=False)
        if initialize and cfg.MODEL.PRETRAIN_CHOICE=='imagenet':
            self.base.load_param(cfg.MODEL.PRETRAIN_PATH,hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
        if not cfg.MODEL.STEM_CONV:
            raise ValueError('This copy uses the original convolutional stride-16 stem')
        self.conv_neck_enabled=bool(cfg.MODEL.CONV_NECK_ENABLED)
        selected=list(cfg.MODEL.CONV_NECK_LAYERS)
        if (not selected or any(type(i) is not int or i<1 or i>12 for i in selected)
            or selected!=sorted(set(selected))):
            raise ValueError('Neck layers must be unique increasing 1-based indices in 1..12')
        self.conv_neck_layers=selected if self.conv_neck_enabled else []
        self.conv_neck_placement=cfg.MODEL.CONV_NECK_PLACEMENT
        if self.conv_neck_placement not in ('pre_attn','post_mlp'):
            raise ValueError('Neck placement must be pre_attn or post_mlp')
        self.detail_enabled=bool(cfg.MODEL.DETAIL_ENABLED)
        self.alpha=float(cfg.MODEL.DETAIL_ALPHA)
        self.detail_dim=int(cfg.MODEL.DETAIL_DIM)
        self.detail_pooling=cfg.MODEL.DETAIL_POOLING
        if not 0<self.alpha<1 or self.detail_dim!=256:
            raise ValueError('Use a nondegenerate mixture and 256D detail output')
        # Added parameters do not advance the baseline RNG stream. Every new
        # module is initialized with 1234 in a forked CPU RNG context.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234)
            self.detail_branch=ViTMatteDetails(768,self.detail_dim,self.detail_pooling)
            self.detail_bottleneck=nn.BatchNorm1d(self.detail_dim)
            self.fused_bottleneck=nn.BatchNorm1d(768+self.detail_dim)
            self.fused_classifier=nn.Linear(768+self.detail_dim,num_classes,bias=False)
            for bn in (self.detail_bottleneck,self.fused_bottleneck):
                nn.init.ones_(bn.weight);nn.init.zeros_(bn.bias);bn.bias.requires_grad_(False)
            nn.init.normal_(self.fused_classifier.weight,std=.001)
        # A separate RNG fork preserves the earlier DCM initialization as well
        # as the original backbone/classifier and caller's random stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234)
            attach_conv_necks(self.base,self.conv_neck_layers,self.conv_neck_placement)
        if not self.detail_enabled:
            for module in (self.detail_branch,self.detail_bottleneck,self.fused_bottleneck,self.fused_classifier):
                module.requires_grad_(False)
        self.descriptor_dim=768+self.detail_dim if self.detail_enabled else 768
        if initialize and cfg.MODEL.PRETRAIN_CHOICE=='self':
            self.load_param(cfg.MODEL.PRETRAIN_PATH)

    def forward(self,x,label=None,cam_label=None,view_label=None,return_components=False,
                return_cnl_features=False,return_training=False,return_details=False,return_maps=False):
        if self.detail_enabled:
            layers,patches=self.base(x,cam_label=cam_label,view_label=view_label,return_spatial=True)
        else:
            layers=self.base(x,cam_label=cam_label,view_label=view_label)
        raw=self.mldc(layers)
        neck=self.bottleneck(raw)
        global_unit=F.normalize(neck.float(),dim=1)
        data=dict(global_raw=raw,global_unit=global_unit,layers=layers)
        if self.detail_enabled:
            detail=self.detail_branch(patches,x,raw,return_maps=return_maps)
            if return_maps:
                detail,data['maps']=detail
            detail_neck=self.detail_bottleneck(detail)
            detail_unit=F.normalize(detail_neck.float(),dim=1)
            fused=fuse_units(global_unit,detail_unit,self.alpha)
            data.update(detail_raw=detail,detail_unit=detail_unit,fused_unit=fused)
        else:
            data['fused_unit']=global_unit
        if self.training:
            scores=self.classifier(self.dropout(neck))
            data['global_scores']=scores
            if self.detail_enabled:
                data['fused_scores']=self.fused_classifier(self.dropout(self.fused_bottleneck(data['fused_unit'])))
            if return_training or return_details:
                return data
            if return_cnl_features:
                return scores,raw,global_unit
            return scores,raw
        if return_details:
            return data
        output=data['fused_unit']
        return (output,raw,layers) if return_components else output

    def architecture(self):
        return dict(variant=VARIANT,detail_enabled=self.detail_enabled,detail_pooling=self.detail_pooling,
            detail_dim=self.detail_dim,alpha=self.alpha,descriptor_dim=self.descriptor_dim,
            cnn_channels=[48,96,192],fusion_channels=[256,128,64,32],rgb_channels=3,
            conv_neck_enabled=self.conv_neck_enabled,conv_neck_layers=self.conv_neck_layers,
            conv_neck_placement=self.conv_neck_placement,conv_neck_hidden=384,
            conv_neck_zero_init=True,conv_neck_cls='patch_only',
            global_layers=[9,10,11,12],cnl_space='global_768',seed=1234)

    def load_param(self,path):
        checkpoint=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        if checkpoint.get('architecture')!=self.architecture():
            raise ValueError('Checkpoint architecture differs from the requested ViTMatte transfer')
        self.load_state_dict(checkpoint['state_dict'],strict=True)
        self.checkpoint_epoch=int(checkpoint['epoch'])

    def load_global_checkpoint(self,path):
        """Explicit diagnostic import; new DCM/Neck parameters stay unchanged."""
        checkpoint=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        if tuple(checkpoint.get('mldc_layers',()))!=(9,10,11,12):
            raise ValueError('Expected the original deep-four checkpoint')
        state={(k[7:] if k.startswith('module.') else k):v for k,v in checkpoint['state_dict'].items()}
        expected={k for k in self.state_dict() if '.conv_neck.' not in k and not k.startswith(
            ('detail_branch.','detail_bottleneck.','fused_bottleneck.','fused_classifier.'))}
        if set(state)!=expected:
            raise ValueError('Global checkpoint does not exactly match the baseline path')
        result=self.load_state_dict(state,strict=False)
        # BatchNorm's legacy loader fills num_batches_tracked itself rather than
        # listing it as missing when a plain state dictionary has no metadata.
        missing={k for k in set(self.state_dict())-expected if not k.endswith('.num_batches_tracked')}
        assert set(result.missing_keys)==missing and not result.unexpected_keys


def make_model(cfg,num_class,camera_num,view_num):
    return MLDCReID(cfg,num_class,camera_num,view_num)
