"""NightReID MLDC+CNL with parallel zero-output-initialized Convpass adapters."""
import math
import torch
from torch import nn
from .base_model import MLDCReID as Baseline,add_mldc_config as baseline_config
from .base_model import configure_head_learning_rates
from .convpass import RectangularConvpass

VARIANT='convpass_mldc_cnl_20261026'


def add_mldc_config(cfg):
    cfg=baseline_config(cfg)
    cfg.MODEL.CONVPASS_ENABLED=True
    cfg.MODEL.CONVPASS_LAYERS=[1,2,3,4]
    cfg.MODEL.CONVPASS_FREEZE_BACKBONE=True
    cfg.MODEL.CONVPASS_DIM=8
    cfg.MODEL.CONVPASS_SCALE=.1
    cfg.TEST.MLDC_VISUALIZE=False
    return cfg


class MLDCReID(Baseline):
    def __init__(self,cfg,num_classes,camera_num,view_num,initialize=True):
        if cfg.SOLVER.SEED!=1234 or cfg.DATASETS.NAMES!='nightreid':
            raise ValueError('Use NightReID and seed1234')
        if (list(cfg.MODEL.MLDC_LAYERS)!=[9,10,11,12] or cfg.MODEL.MLDC_READOUT!='mldc'
            or cfg.MODEL.MLDC_TRAIN_FEATURE!='mldc' or cfg.TEST.MLDC_RETRIEVAL!='mldc'
            or cfg.TEST.NECK_FEAT!='after' or cfg.TEST.FEAT_NORM!='yes'
            or list(cfg.MODEL.STRIDE_SIZE)!=[16,16] or not cfg.MODEL.STEM_CONV
            or cfg.MODEL.SIE_CAMERA or cfg.MODEL.SIE_VIEW):
            raise ValueError('Use original stride16 stem, SIE-off and deep4 BN-after MLDC')
        layers=list(cfg.MODEL.CONVPASS_LAYERS)
        if not layers or layers!=sorted(set(layers)) or any(type(i) is not int or i<1 or i>12 for i in layers):
            raise ValueError('Convpass layers must be increasing unique indices in 1..12')
        if cfg.MODEL.CONVPASS_DIM!=8:
            raise ValueError('Use the authors\' dimension-8 Convpass module')
        scale=float(cfg.MODEL.CONVPASS_SCALE)
        if not math.isfinite(scale) or scale<=0:raise ValueError('Convpass scale must be positive')
        super().__init__(cfg,num_classes,camera_num,view_num,initialize=False)
        if initialize and cfg.MODEL.PRETRAIN_CHOICE=='imagenet':
            self.base.load_param(cfg.MODEL.PRETRAIN_PATH,hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
        self.convpass_enabled=bool(cfg.MODEL.CONVPASS_ENABLED)
        self.convpass_layers=layers if self.convpass_enabled else []
        self.freeze_backbone=bool(cfg.MODEL.CONVPASS_FREEZE_BACKBONE)
        if not self.convpass_enabled and self.freeze_backbone:
            raise ValueError('The disabled baseline control uses joint backbone training')
        self.convpass_scale=scale
        self.initialization_report=[]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234)
            for layer in self.convpass_layers:
                block=self.base.blocks[layer-1]
                block.adapter_attn=RectangularConvpass(dim=8,xavier_init=False)
                block.adapter_mlp=RectangularConvpass(dim=8,xavier_init=False)
                block.convpass_scale=scale
                self.initialization_report.append(dict(layer=layer,sites=['LN1/attention parallel','LN2/FFN parallel'],
                    original_parameters='retained unchanged',down='Xavier',conv='center identity',up='zero'))
        if self.freeze_backbone:
            self.base.requires_grad_(False)
            for layer in self.convpass_layers:
                block=self.base.blocks[layer-1]
                block.adapter_attn.requires_grad_(True)
                block.adapter_mlp.requires_grad_(True)
        if initialize and cfg.MODEL.PRETRAIN_CHOICE=='self':self.load_param(cfg.MODEL.PRETRAIN_PATH)

    def train(self,mode=True):
        super().train(mode)
        if self.freeze_backbone:
            # Plain ViT in the original paper has no BN. This ReID backbone
            # has a convolutional stem: freeze its running statistics too.
            # Leave original stochastic depth and adapter dropout in train mode.
            for module in self.base.modules():
                if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
        return self

    def architecture(self):
        return dict(variant=VARIANT,convpass_enabled=self.convpass_enabled,
            convpass_layers=self.convpass_layers,freeze_backbone=self.freeze_backbone,
            convpass_dim=8,convpass_scale=self.convpass_scale,
            convpass_sites=['attention_parallel','ffn_parallel'],convpass_init='official_zero_up',
            convpass_cls='separate_1x1_image',convpass_dropout=.1,
            original_ln_ffn_cls='retained',frozen_stem_bn_stats=self.freeze_backbone,
            global_layers=[9,10,11,12],mldc_temperature=self.mldc.temperature,
            descriptor_dim=768,patch_grid=[16,8],dataset='nightreid',seed=1234)

    def load_param(self,path):
        checkpoint=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        if checkpoint.get('architecture')!=self.architecture():
            raise ValueError('Checkpoint Convpass architecture/training policy differs')
        self.load_state_dict(checkpoint['state_dict'],strict=True)
        self.checkpoint_epoch=int(checkpoint['epoch'])

    def load_global_checkpoint(self,path):
        """Diagnostic import only; leave all newly created adapters untouched."""
        checkpoint=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        if tuple(checkpoint.get('mldc_layers',()))!=(9,10,11,12):
            raise ValueError('Expected the plain deep4 MLDC/CNL baseline checkpoint')
        state={(k[7:] if k.startswith('module.') else k):v for k,v in checkpoint['state_dict'].items()}
        expected={k for k in self.state_dict() if '.adapter_attn.' not in k and '.adapter_mlp.' not in k}
        if set(state)!=expected:raise ValueError('Baseline weights do not match the original path')
        result=self.load_state_dict(state,strict=False)
        assert set(result.missing_keys)==set(self.state_dict())-expected and not result.unexpected_keys


def make_model(cfg,num_class,camera_num,view_num):
    model=MLDCReID(cfg,num_class,camera_num,view_num)
    print('Convpass layers={}, frozen_backbone={}, descriptor=768, seed1234'.format(
        model.convpass_layers,model.freeze_backbone))
    return model
