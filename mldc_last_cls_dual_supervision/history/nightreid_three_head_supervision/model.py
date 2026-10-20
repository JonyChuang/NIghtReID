"""Shared ViT, two source heads and a supervised NightReID-style concatenation."""
import math
import torch
from torch import nn
from torch.nn import functional as F

VARIANT='mldc_fixed4_lastln_nightreid_three_heads_20261019'
BRANCHES=('fusion','last','concat')


def add_config(cfg):
    cfg.MODEL.GLOBAL_LAYERS=[9,10,11,12]
    cfg.MODEL.GLOBAL_FUSION='fixed_equal'
    cfg.MODEL.LAST_LOSS_WEIGHT=.3
    cfg.MODEL.CONCAT_LOSS_WEIGHT=1.
    cfg.SOLVER.MLDC_HEAD_LR_FACTOR=5.
    cfg.TEST.DUAL_QUERY_CHUNK=32
    return cfg


def config_from_dump(base_cfg,text):
    cfg=add_config(base_cfg.clone());stored=type(cfg).load_cfg(text)
    # The installed initial version used equal supervision without this key.
    if 'LAST_LOSS_WEIGHT' not in stored.MODEL:cfg.MODEL.LAST_LOSS_WEIGHT=.5
    stored.MODEL.DEVICE_ID=repr(str(stored.MODEL.DEVICE_ID))
    cfg.merge_from_other_cfg(stored)
    return cfg


def loss_weights(cfg):
    last=float(cfg.MODEL.LAST_LOSS_WEIGHT)
    if not math.isfinite(last) or not 0.<last<1.:
        raise ValueError('LAST_LOSS_WEIGHT is the last-branch fraction within the two source losses, in (0,1)')
    return 1.-last,last


def concat_loss_weight(cfg):
    weight=float(cfg.MODEL.CONCAT_LOSS_WEIGHT)
    if not math.isfinite(weight) or weight<=0.:
        raise ValueError('CONCAT_LOSS_WEIGHT must be finite and positive')
    return weight


def retrieval_features(raw,neck,neck_feat):
    """Select BN-before/after, then normalize each complete descriptor once."""
    if neck_feat not in ('before','after'):raise ValueError('Unknown NECK_FEAT')
    features=raw if neck_feat=='before' else neck
    return {name:F.normalize(features[name].float(),dim=-1,eps=1e-12) for name in BRANCHES}


class DualMLDCReID(nn.Module):
    def __init__(self,cfg,num_classes,cameras=0,views=0,initialize=True):
        super().__init__()
        from .backbones.vit_pytorch_mldc import vit_base_patch16_224_TransReID
        if list(cfg.MODEL.GLOBAL_LAYERS)!=[9,10,11,12] or cfg.MODEL.GLOBAL_FUSION!='fixed_equal':
            raise ValueError('Use fixed arithmetic mean of raw post-block CLS9-12')
        self.fusion_loss_weight,self.last_loss_weight=loss_weights(cfg)
        self.concat_loss_weight=concat_loss_weight(cfg)
        self.neck_feat=cfg.TEST.NECK_FEAT
        if (cfg.MODEL.PRETRAIN_CHOICE!='imagenet' or cfg.MODEL.PRETRAIN_PATH!='pretrain_model/vit_base_ics_cfs_lup.pth'
            or cfg.MODEL.TRANSFORMER_TYPE!='vit_base_patch16_224_TransReID'
            or cfg.MODEL.ID_LOSS_TYPE!='softmax' or cfg.MODEL.JPM or cfg.MODEL.USE_DCE
            or cfg.MODEL.QNORM_PRIOR or cfg.MODEL.QNORM_GUIDE or cfg.MODEL.GEM_POOLING
            or cfg.MODEL.DIST_TRAIN or cfg.MODEL.REDUCE_FEAT_DIM or self.neck_feat not in ('before','after')):
            raise ValueError('Use the matched plain SSL initialization and BN-before/after retrieval')
        # Match the fixed-equal MLDC control's backbone/head RNG consumption.
        torch.randn(12)
        self.base=vit_base_patch16_224_TransReID(img_size=cfg.INPUT.SIZE_TRAIN,stride_size=cfg.MODEL.STRIDE_SIZE,
            camera=cameras if cfg.MODEL.SIE_CAMERA else 0,view=views if cfg.MODEL.SIE_VIEW else 0,
            sie_xishu=cfg.MODEL.SIE_COE,stem_conv=cfg.MODEL.STEM_CONV,
            drop_path_rate=cfg.MODEL.DROP_PATH,drop_rate=cfg.MODEL.DROP_OUT,attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if initialize:self.base.load_param(cfg.MODEL.PRETRAIN_PATH,hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
        # Only unused output modules are excluded; final LayerNorm IS active/trainable.
        for module in (self.base.fc,self.base.gem):module.requires_grad_(False)
        self.base.norm.requires_grad_(True)
        self.register_buffer('fusion_weights',torch.tensor([0.]*8+[.25]*4))
        self.bottleneck=nn.BatchNorm1d(768)
        self.bottleneck.bias.requires_grad_(False)  # Original ReID BN convention.
        self.classifier=nn.Linear(768,num_classes,bias=False)
        nn.init.normal_(self.classifier.weight,std=.001)
        self.last_bottleneck=nn.BatchNorm1d(768)
        self.last_bottleneck.bias.requires_grad_(False)
        self.last_classifier=nn.Linear(768,num_classes,bias=False)
        nn.init.normal_(self.last_classifier.weight,std=.001)
        self.concat_bottleneck=nn.BatchNorm1d(1536)
        self.concat_bottleneck.bias.requires_grad_(False)
        self.concat_classifier=nn.Linear(1536,num_classes,bias=False)
        nn.init.normal_(self.concat_classifier.weight,std=.001)
        self.dropout=nn.Dropout(cfg.MODEL.DROPOUT_RATE)

    def forward(self,images,label=None,cam_label=None,view_label=None,diagnostics=False):
        layers=self.base(images,cam_label=cam_label,view_label=view_label)
        fusion=layers[8:12].mean(0)
        last=self.base.norm(layers[-1])
        # NightReID concatenates BN-before descriptors, without branch-wise L2.
        concat=torch.cat((fusion,last),dim=-1)
        gneck=self.bottleneck(fusion);cneck=self.last_bottleneck(last)
        zneck=self.concat_bottleneck(concat)
        raw=dict(fusion=fusion,last=last,concat=concat)
        neck=dict(fusion=gneck,last=cneck,concat=zneck)
        retrieval=retrieval_features(raw,neck,self.neck_feat)
        if self.training or diagnostics:
            result=dict(raw=raw,neck=neck,retrieval=retrieval,layers=layers)
            if self.training:
                result['scores']=dict(fusion=self.classifier(self.dropout(gneck)),last=self.last_classifier(self.dropout(cneck)),
                    concat=self.concat_classifier(self.dropout(zneck)))
            return result
        return retrieval['concat']

    def architecture(self):
        return dict(variant=VARIANT,shared_backbone=True,global_layers=[9,10,11,12],global_fusion='fixed_equal',
            global_weights=[.25]*4,last_branch='raw CLS12 -> pretrained trainable final affine LayerNorm',
            identity='three independent BNNecks/classifiers, including a 1536D concatenation head',
            triplet='batch-hard independently on raw fusion, last-LN and their raw concatenation',
            fusion_loss_weight=self.fusion_loss_weight,last_loss_weight=self.last_loss_weight,
            concat_loss_weight=self.concat_loss_weight,
            loss=f'{self.fusion_loss_weight:g}*(ID_fusion+Triplet_fusion)+{self.last_loss_weight:g}*(ID_last+Triplet_last)+{self.concat_loss_weight:g}*(ID_concat+Triplet_concat)',
            concatenation='cat(raw fusion, last-LN), before branch BN and without branch-wise L2',
            retrieval=f'BN-{self.neck_feat}; L2 each complete descriptor; concat has one global L2',descriptor_dim=1536,
            neck_feat=self.neck_feat,reference='https://github.com/msm8976/NightReID',
            single_branch_dim=768,evaluate_branches=list(BRANCHES),joint_from_epoch=1,
            active_backbone_frozen=False,patch_enhancement=False,spatial_partition=False)


def objectives(data,labels,cfg):
    from loss.triplet_loss import euclidean_dist,hard_example_mining
    with torch.autocast(device_type=labels.device.type,enabled=False):
        terms={}
        for name in BRANCHES:
            terms[name+'_id']=F.cross_entropy(data['scores'][name].float(),labels)
            feature=data['raw'][name].float()
            positive,negative=hard_example_mining(euclidean_dist(feature,feature),labels)
            terms[name+'_triplet']=F.softplus(positive-negative).mean() if cfg.MODEL.NO_MARGIN else F.relu(positive-negative+cfg.SOLVER.MARGIN).mean()
        wg,wc=loss_weights(cfg)
        loss=wg*(cfg.MODEL.ID_LOSS_WEIGHT*terms['fusion_id']+cfg.MODEL.TRIPLET_LOSS_WEIGHT*terms['fusion_triplet'])+wc*(cfg.MODEL.ID_LOSS_WEIGHT*terms['last_id']+cfg.MODEL.TRIPLET_LOSS_WEIGHT*terms['last_triplet'])
        loss=loss+concat_loss_weight(cfg)*(cfg.MODEL.ID_LOSS_WEIGHT*terms['concat_id']+cfg.MODEL.TRIPLET_LOSS_WEIGHT*terms['concat_triplet'])
        if not torch.isfinite(loss).item():raise FloatingPointError('Non-finite dual-supervision loss')
    return loss,terms
