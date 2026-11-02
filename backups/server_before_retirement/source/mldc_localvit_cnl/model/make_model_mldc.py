"""NightReID MLDC+CNL with official LocalViT FFNs in Blocks 1--4 only."""
import torch
from .base_model import MLDCReID as Baseline,add_mldc_config as baseline_config
from .base_model import configure_head_learning_rates
from .localvit_official import LocalityFeedForward

VARIANT='localvit_first4_mldc_cnl_20261025'


def add_mldc_config(cfg):
    cfg=baseline_config(cfg)
    cfg.MODEL.LOCALVIT_ENABLED=True
    cfg.MODEL.LOCALVIT_LAYERS=[1,2,3,4]
    cfg.TEST.MLDC_VISUALIZE=False
    return cfg


class MLDCReID(Baseline):
    def __init__(self,cfg,num_classes,camera_num,view_num,initialize=True):
        if cfg.SOLVER.SEED!=1234 or cfg.DATASETS.NAMES!='nightreid':
            raise ValueError('This experiment fixes NightReID and seed1234')
        if (list(cfg.MODEL.LOCALVIT_LAYERS)!=[1,2,3,4]
            or list(cfg.MODEL.MLDC_LAYERS)!=[9,10,11,12]
            or cfg.MODEL.MLDC_READOUT!='mldc' or cfg.MODEL.MLDC_TRAIN_FEATURE!='mldc'
            or cfg.TEST.MLDC_RETRIEVAL!='mldc' or cfg.TEST.NECK_FEAT!='after'
            or cfg.TEST.FEAT_NORM!='yes' or list(cfg.MODEL.STRIDE_SIZE)!=[16,16]
            or not cfg.MODEL.STEM_CONV or cfg.MODEL.SIE_CAMERA or cfg.MODEL.SIE_VIEW):
            raise ValueError('Use first4 LocalViT, deep4 MLDC, original stride16 stem and SIE-off protocol')
        super().__init__(cfg,num_classes,camera_num,view_num,initialize=False)
        # Load original SSL keys before replacing any FFNs.
        if initialize and cfg.MODEL.PRETRAIN_CHOICE=='imagenet':
            self.base.load_param(cfg.MODEL.PRETRAIN_PATH,hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
        self.localvit_enabled=bool(cfg.MODEL.LOCALVIT_ENABLED)
        self.localvit_layers=[1,2,3,4] if self.localvit_enabled else []
        self.initialization_report=[]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234)
            for layer in self.localvit_layers:
                block=self.base.blocks[layer-1]
                dim=block.mlp.fc1.in_features
                hidden=block.mlp.fc1.out_features
                assert hidden==4*dim
                # SE has four squeezed channels, following the lightweight
                # r=embedding_dim setting in the authors' LocalViT models.
                local=LocalityFeedForward(dim,dim,1,expand_ratio=4,act='hs+se',reduction=dim)
                local.apply(self.base._init_weights)
                with torch.no_grad():
                    local.conv[0].weight.copy_(block.mlp.fc1.weight[:,:,None,None])
                    local.conv[-2].weight.copy_(block.mlp.fc2.weight[:,:,None,None])
                self.initialization_report.append(dict(layer=layer,
                    mapped=['mlp.fc1.weight -> local_ffn.conv.0.weight',
                            'mlp.fc2.weight -> local_ffn.conv.7.weight'],
                    removed=['mlp.fc1.bias','mlp.fc2.bias','norm2.weight','norm2.bias'],
                    new=['depthwise convolution','BatchNorm2d','h-swish','SE(hidden=4)']))
                block.local_ffn=local
                # Remove replaced modules, rather than leave dead trainable
                # parameters or an extra original-MLP path in the optimizer.
                del block.mlp
                del block.norm2
        if initialize and cfg.MODEL.PRETRAIN_CHOICE=='self':
            self.load_param(cfg.MODEL.PRETRAIN_PATH)

    def architecture(self):
        return dict(variant=VARIANT,localvit_enabled=self.localvit_enabled,
            localvit_layers=self.localvit_layers,localvit_expansion=4,
            localvit_activation='hs+se',localvit_se_reduction=768,localvit_se_hidden=4,
            localvit_norm='BatchNorm2d',localvit_cls='bypass_complete_local_ffn',
            localvit_residual='official_patch_residual_no_extra_droppath',
            ssl_ffn_mapping='linear_weights_to_1x1; discard_replaced_bias_and_norm2',
            patch_grid=[16,8],input_size=[256,128],global_layers=[9,10,11,12],
            mldc_temperature=self.mldc.temperature,descriptor_dim=768,
            cnl_space='global_bn_after_l2_768',dataset='nightreid',seed=1234)

    def load_param(self,path):
        checkpoint=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        if checkpoint.get('architecture')!=self.architecture():
            raise ValueError('Checkpoint does not match this first-four LocalViT architecture')
        self.load_state_dict(checkpoint['state_dict'],strict=True)
        self.checkpoint_epoch=int(checkpoint['epoch'])


def make_model(cfg,num_class,camera_num,view_num):
    model=MLDCReID(cfg,num_class,camera_num,view_num)
    print('LocalViT layers={}, MLDC layers={}, descriptor=768, seed1234'.format(
        model.localvit_layers,model.mldc.selected_layers))
    return model
