"""Frozen deep-four CLS anchor with independently supervised patch-GRM readout."""
from pathlib import Path
import torch
from torch import nn
from .backbones.vit_pytorch_mldc import vit_base_patch16_224_TransReID
from .mldc import MLDC
from .patch_compensation import PatchCompensation

EXPERIMENT='mldc_patch_direct_grm_v2_20261016'
PACKAGE=Path(__file__).resolve().parents[1]
ANCHOR_PREFIXES=('base.','mldc.','bottleneck.','classifier.')


def add_mldc_config(cfg):
    cfg.MODEL.MLDC_READOUT='mldc'
    cfg.MODEL.MLDC_TEMPERATURE=0.9
    cfg.MODEL.MLDC_LAYERS=[9,10,11,12]
    cfg.MODEL.MLDC_INIT='uniform'
    cfg.MODEL.MLDC_TRAIN_FEATURE='patch_direct_grm'
    cfg.MODEL.MLDC_DEEP_INIT_MASS=0.75
    cfg.MODEL.PATCH_LAYERS=list(range(1,13))
    cfg.MODEL.PATCH_HIDDEN_DIM=192
    cfg.MODEL.PATCH_HEADS=4
    cfg.MODEL.PATCH_USE_NULL=False
    cfg.MODEL.PATCH_ZERO_OUTPUT=False
    cfg.MODEL.GRM_MLP_HIDDEN=256
    cfg.MODEL.GRM_MAX_GATE=0.1
    cfg.MODEL.GRM_INITIAL_GATE=0.01
    cfg.MODEL.PATCH_BASELINE_CHECKPOINT=str(PACKAGE/'initialization/deep4_uniform_epoch120.pth')
    cfg.SOLVER.MLDC_HEAD_LR_FACTOR=5.0
    cfg.SOLVER.MLDC_WEIGHT_LR_FACTOR=1.0
    cfg.SOLVER.PATCH_WARMUP_EPOCHS=5
    cfg.SOLVER.PATCH_LOCAL_LOSS_WEIGHT=0.1
    cfg.TEST.MLDC_VISUALIZE=True
    cfg.TEST.MLDC_VIS_SAMPLES=6
    cfg.TEST.MLDC_VIS_PERIOD=10
    cfg.TEST.MLDC_RETRIEVAL='mldc'
    return cfg


def configure_head_learning_rates(cfg, model, optimizer):
    heads={id(p) for n,p in model.named_parameters() if p.requires_grad and n.startswith('local_classifier.')}
    for group in optimizer.param_groups:
        ids={id(p) for p in group['params']}
        if ids & heads:
            if not ids <= heads:
                raise RuntimeError('Optimizer mixes local classifier and other parameters')
            group['lr']*=cfg.SOLVER.MLDC_HEAD_LR_FACTOR


class MLDCReID(nn.Module):
    def __init__(self,cfg,num_classes,camera_num,view_num):
        super().__init__()
        if cfg.MODEL.NAME!='transformer' or cfg.MODEL.TRANSFORMER_TYPE!='vit_base_patch16_224_TransReID':
            raise ValueError('This experiment uses the completed Night600 ViT-Base anchor')
        if any((cfg.MODEL.JPM,cfg.MODEL.GEM_POOLING,cfg.MODEL.USE_DCE,cfg.MODEL.QNORM_PRIOR,
                cfg.MODEL.QNORM_GUIDE,cfg.MODEL.DIST_TRAIN,cfg.MODEL.REDUCE_FEAT_DIM)):
            raise ValueError('Use the original plain single-GPU backbone')
        if (cfg.MODEL.ID_LOSS_TYPE!='softmax' or cfg.MODEL.PRETRAIN_CHOICE!='self'
                or cfg.MODEL.MLDC_READOUT!='mldc' or list(cfg.MODEL.MLDC_LAYERS)!=[9,10,11,12]
                or cfg.MODEL.MLDC_TRAIN_FEATURE!='patch_direct_grm'
                or cfg.TEST.MLDC_RETRIEVAL!='mldc' or cfg.TEST.NECK_FEAT not in ('before','after')
                or cfg.SOLVER.MLDC_WEIGHT_LR_FACTOR!=1.0):
            raise ValueError('Use the patch-direct-GRM configuration and completed deep-four initialization')
        if cfg.MODEL.PATCH_USE_NULL or cfg.MODEL.PATCH_ZERO_OUTPUT:
            raise ValueError('GRM replaces the null option and zero local output')
        self.warmup_epochs=int(cfg.SOLVER.PATCH_WARMUP_EPOCHS)
        if self.warmup_epochs < 1 or self.warmup_epochs >= cfg.SOLVER.MAX_EPOCHS:
            raise ValueError('Local warmup must be positive and shorter than the complete run')
        self.register_buffer('_training_epoch',torch.tensor(0,dtype=torch.long))
        self.readout,self.retrieval='mldc','mldc'
        self.train_feature='patch_direct_grm'
        self.checkpoint_epoch,self.baseline_epoch=0,120
        self.baseline_checkpoint=str(cfg.MODEL.PATCH_BASELINE_CHECKPOINT)
        self.mldc=MLDC(12,cfg.MODEL.MLDC_TEMPERATURE,init=cfg.MODEL.MLDC_INIT,
                       selected_layers=cfg.MODEL.MLDC_LAYERS)
        self.base=vit_base_patch16_224_TransReID(
            img_size=cfg.INPUT.SIZE_TRAIN,stride_size=cfg.MODEL.STRIDE_SIZE,
            camera=camera_num if cfg.MODEL.SIE_CAMERA else 0,
            view=view_num if cfg.MODEL.SIE_VIEW else 0,sie_xishu=cfg.MODEL.SIE_COE,
            stem_conv=cfg.MODEL.STEM_CONV,drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate=cfg.MODEL.DROP_OUT,attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        dim=self.base.in_planes
        self.bottleneck=nn.BatchNorm1d(dim)
        self.classifier=nn.Linear(dim,num_classes,bias=False)
        self.neck_feat=cfg.TEST.NECK_FEAT
        self.patch_compensation=PatchCompensation(dim,cfg.MODEL.PATCH_HIDDEN_DIM,cfg.MODEL.PATCH_HEADS,
                                                 cfg.MODEL.PATCH_LAYERS,cfg.MODEL.GRM_MLP_HIDDEN,
                                                 cfg.MODEL.GRM_MAX_GATE,cfg.MODEL.GRM_INITIAL_GATE)
        self.local_bottleneck=nn.BatchNorm1d(dim)
        nn.init.ones_(self.local_bottleneck.weight)
        nn.init.zeros_(self.local_bottleneck.bias)
        self.local_bottleneck.bias.requires_grad_(False)
        self.local_classifier=nn.Linear(dim,num_classes,bias=False)
        nn.init.normal_(self.local_classifier.weight,std=0.001)
        self.dropout=nn.Dropout(cfg.MODEL.DROPOUT_RATE)
        allow=Path(cfg.MODEL.PRETRAIN_PATH).resolve()==Path(self.baseline_checkpoint).resolve()
        self.load_param(cfg.MODEL.PRETRAIN_PATH,allow_baseline=allow)
        for module in (self.base,self.mldc,self.bottleneck,self.classifier):
            module.requires_grad_(False)
        self.train(self.training)

    @property
    def stage(self):
        return 'local_warmup' if int(self._training_epoch.item())<=self.warmup_epochs else 'joint_fusion'

    def set_epoch(self,epoch):
        if type(epoch) is not int or epoch<0:
            raise ValueError('Epoch must be a non-negative integer')
        self._training_epoch.fill_(epoch)

    def train(self,mode=True):
        super().train(mode)
        for module in (self.base,self.mldc,self.bottleneck,self.classifier):
            module.eval()
        return self

    def forward(self,x,label=None,cam_label=None,view_label=None,return_components=False,
                return_patch_diagnostics=False):
        if return_patch_diagnostics and self.training:
            raise ValueError('Patch diagnostics are evaluation-only')
        layers,patches=self.base(x,cam_label=cam_label,view_label=view_label,
                                 return_patches=True,patch_layers=self.patch_compensation.layers)
        anchor=self.mldc(layers)
        stage=self.stage
        patch=self.patch_compensation(anchor,patches,fusion_enabled=stage=='joint_fusion',
                                      return_attention=return_patch_diagnostics)
        feature=anchor+patch['delta']
        neck=self.bottleneck(feature)
        if self.training:
            local_neck=self.local_bottleneck(patch['local'])
            return {'scores':self.classifier(self.dropout(neck)),'feature':feature,
                    'local_scores':self.local_classifier(self.dropout(local_neck)),
                    'local_feature':patch['local'],'stage':stage,'gate':patch['gate']}
        output=neck if self.neck_feat=='after' else feature
        if return_patch_diagnostics:
            return {'output':output,'raw':feature,'anchor':anchor,'cls_features':layers,
                    'local_bn':self.local_bottleneck(patch['local']),'stage':stage,**patch}
        return (output,feature,layers) if return_components else output

    def readout_weights(self):
        return self.mldc.normalized_weights()

    def checkpoint_metadata(self):
        return {'experiment':EXPERIMENT,'stage':self.stage,'patch_architecture':self.patch_compensation.architecture(),
                'local_warmup_epochs':self.warmup_epochs,'baseline_checkpoint':self.baseline_checkpoint,
                'baseline_epoch':self.baseline_epoch,'training_epoch':int(self._training_epoch.item())}

    def load_param(self,path,allow_baseline=False):
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        if (tuple(checkpoint.get('mldc_layers',()))!=self.mldc.selected_layers
                or checkpoint.get('readout')!='mldc' or checkpoint.get('temperature')!=self.mldc.temperature):
            raise ValueError('Checkpoint does not match the original deep-four CLS anchor')
        state=checkpoint.get('state_dict',checkpoint.get('model',checkpoint))
        state={k[7:] if k.startswith('module.') else k:v for k,v in state.items()}
        if checkpoint.get('experiment')==EXPERIMENT:
            if (checkpoint.get('patch_architecture')!=self.patch_compensation.architecture()
                    or checkpoint.get('train_feature')!=self.train_feature
                    or checkpoint.get('local_warmup_epochs')!=self.warmup_epochs):
                raise ValueError('GRM checkpoint architecture or stage policy differs')
            epoch=int(checkpoint.get('epoch',-1))
            expected='local_warmup' if epoch<=self.warmup_epochs else 'joint_fusion'
            if (epoch<0 or checkpoint.get('training_epoch')!=epoch or checkpoint.get('stage')!=expected
                    or '_training_epoch' not in state or int(state['_training_epoch'].item())!=epoch):
                raise ValueError('Checkpoint epoch and saved fusion stage disagree')
            self.load_state_dict(state,strict=True)
            self.checkpoint_epoch=epoch
            self.baseline_epoch=int(checkpoint.get('baseline_epoch',120))
        else:
            if (not allow_baseline or checkpoint.get('train_feature')!='mldc'
                    or any(k.startswith(('patch_compensation.','local_')) for k in state)):
                raise ValueError('Legacy patch-v1 checkpoints are incompatible; use the configured deep-four baseline or a GRM-v2 checkpoint')
            current=self.state_dict()
            expected={k for k in current if k.startswith(ANCHOR_PREFIXES)}
            if set(state)!=expected:
                raise ValueError('Baseline checkpoint contains missing or unexpected anchor tensors')
            current.update(state)
            self.load_state_dict(current,strict=True)
            self.baseline_epoch=int(checkpoint.get('epoch',0))
            self.checkpoint_epoch=0
            self.set_epoch(0)


def make_model(cfg,num_class,camera_num,view_num):
    model=MLDCReID(cfg,num_class,camera_num,view_num)
    print('{}: CLS={}, patch={}, stage={}, local warmup={}, max gate={}, descriptor=768'.format(
        EXPERIMENT,model.mldc.selected_layers,model.patch_compensation.layers,model.stage,
        model.warmup_epochs,model.patch_compensation.grm.max_gate))
    return model
