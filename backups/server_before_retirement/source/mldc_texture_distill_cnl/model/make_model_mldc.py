import torch
from torch import nn
from yacs.config import CfgNode
from .base_model import MLDCReID as Baseline,add_mldc_config as add_baseline_config
from .base_model import configure_head_learning_rates as configure_baseline_rates
from .texture_teacher import TextureTeacher,distillation_terms


def add_mldc_config(cfg):
    cfg=add_baseline_config(cfg)
    cfg.TEXTURE=CfgNode()
    cfg.TEXTURE.ENABLED=True
    cfg.TEXTURE.BANK='texture'
    cfg.TEXTURE.LAYER=4
    cfg.TEXTURE.DIM=128
    cfg.TEXTURE.TEACHER_PATH=''
    cfg.TEXTURE.TEACHER_EPOCHS=40
    cfg.TEXTURE.TEACHER_LR=0.01
    cfg.TEXTURE.WEIGHT=0.1
    cfg.TEXTURE.RAMP_EPOCHS=20
    cfg.TEXTURE.FEATURE_WEIGHT=0.
    cfg.TEXTURE.RELATION_WEIGHT=1.
    cfg.TEXTURE.ATTENTION_WEIGHT=0.1
    cfg.TEXTURE.ADAPTER_LR_FACTOR=5.
    return cfg


def configure_head_learning_rates(cfg,model,optimizer):
    configure_baseline_rates(cfg,model,optimizer)
    adapter={id(p) for name,p in model.named_parameters() if name.startswith('texture_adapter.') and p.requires_grad}
    for group in optimizer.param_groups:
        ids={id(p) for p in group['params']}
        if ids & adapter:
            if not ids<=adapter: raise ValueError('Mixed backbone and texture-adapter optimizer group')
            group['lr']*=cfg.TEXTURE.ADAPTER_LR_FACTOR


class MLDCReID(Baseline):
    def __init__(self,cfg,num_classes,camera_num,view_num,initialize=True,teacher_override=None):
        super().__init__(cfg,num_classes,camera_num,view_num,
            initialize=initialize and cfg.MODEL.PRETRAIN_CHOICE=='imagenet')
        self.texture_enabled=bool(cfg.TEXTURE.ENABLED)
        self.texture_layer=int(cfg.TEXTURE.LAYER)
        self.texture_settings=cfg.TEXTURE.clone()
        self.texture_teacher=None; self.texture_adapter=None; self._texture_requested=False
        self._texture_tokens=None
        if self.texture_enabled:
            if (list(cfg.INPUT.SIZE_TRAIN)!=[256,128] or list(cfg.INPUT.SIZE_TEST)!=[256,128]
                    or list(cfg.MODEL.STRIDE_SIZE)!=[16,16] or not cfg.MODEL.STEM_CONV):
                raise ValueError('The aligned teacher uses the 256x128 CNN-stem, stride16 protocol')
            if not 1<=self.texture_layer<=11: raise ValueError('Distill an intermediate patch representation')
            if cfg.MODEL.SIE_CAMERA or cfg.MODEL.SIE_VIEW: raise ValueError('Keep the matched SIE-off protocol')
            # Auxiliary initialization must not change the original backbone/head RNG stream.
            with torch.random.fork_rng(devices=[]):
                self.texture_teacher=teacher_override or TextureTeacher(num_classes,cfg.TEXTURE.BANK,
                    (cfg.INPUT.SIZE_TRAIN[0]//16,cfg.INPUT.SIZE_TRAIN[1]//16),cfg.TEXTURE.DIM)
                self.texture_adapter=nn.Sequential(nn.LayerNorm(768,eps=1e-6),nn.Linear(768,cfg.TEXTURE.DIM,bias=False))
            if initialize and cfg.MODEL.PRETRAIN_CHOICE=='imagenet' and teacher_override is None:
                self.texture_teacher.load_trained(cfg.TEXTURE.TEACHER_PATH,cfg.TEXTURE.TEACHER_EPOCHS)
            self.texture_teacher.requires_grad_(False).eval()
            self.base.blocks[self.texture_layer-1].register_forward_hook(self._capture_patches)
        if initialize and cfg.MODEL.PRETRAIN_CHOICE=='self': self.load_param(cfg.MODEL.PRETRAIN_PATH)

    def _capture_patches(self,module,args,result):
        if self._texture_requested: self._texture_tokens=result[:,1:]

    def train(self,mode=True):
        super().train(mode)
        if self.texture_teacher is not None: self.texture_teacher.eval()
        return self

    def forward(self,x,label=None,cam_label=None,view_label=None,return_components=False,
                return_cnl_features=False,return_distillation=False):
        requested=self.training and return_distillation and self.texture_enabled
        self._texture_requested=requested
        try:
            result=super().forward(x,label,cam_label,view_label,return_components,return_cnl_features)
        finally:
            self._texture_requested=False
        if not self.training or not return_distillation: return result
        if not return_cnl_features: raise ValueError('Training distillation returns the original three objectives plus auxiliary terms')
        if not self.texture_enabled:
            zero=result[1].sum()*0
            return (*result,dict(feature=zero,relation=zero,attention=zero))
        tokens=self._texture_tokens; self._texture_tokens=None
        if tokens is None: raise RuntimeError('Missing selected-layer patch features')
        height,width=x.shape[-2]//16,x.shape[-1]//16
        with torch.cuda.amp.autocast(enabled=False):
            student=self.texture_adapter(tokens.float()).transpose(1,2).reshape(len(x),-1,height,width)
            with torch.no_grad(): teacher=self.texture_teacher(x.detach().float())['spatial']
            terms=distillation_terms(student,teacher)
        return (*result,terms)

    def distillation_loss(self,terms,epoch):
        settings=self.texture_settings
        if not self.texture_enabled: return terms['feature']*0
        ramp=min(1.,max(0.,float(epoch)/max(int(settings.RAMP_EPOCHS),1)))
        return float(settings.WEIGHT)*ramp*(settings.FEATURE_WEIGHT*terms['feature']+
            settings.RELATION_WEIGHT*terms['relation']+settings.ATTENTION_WEIGHT*terms['attention'])

    def architecture(self):
        s=self.texture_settings
        return dict(variant='texture_distill_mldc_cnl_20261026',descriptor_dim=768,
            texture_enabled=self.texture_enabled,texture_bank=s.BANK,texture_layer=self.texture_layer,
            texture_dim=int(s.DIM),feature_weight=float(s.FEATURE_WEIGHT),
            relation_weight=float(s.RELATION_WEIGHT),attention_weight=float(s.ATTENTION_WEIGHT),
            distillation_weight=float(s.WEIGHT),ramp_epochs=int(s.RAMP_EPOCHS),
            inference='original MLDC BN-after only',teacher_training='offline identity+triplet, frozen SSL stem')

    def inference_state_dict(self):
        return {k:v for k,v in self.state_dict().items()
            if not k.startswith(('texture_teacher.','texture_adapter.'))}

    def load_param(self,path):
        checkpoint=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        if checkpoint.get('architecture')!=self.architecture():
            raise ValueError('Checkpoint texture-distillation architecture differs')
        self.load_state_dict(checkpoint['state_dict'],strict=True)
        self.checkpoint_epoch=int(checkpoint['epoch'])
        if self.texture_teacher is not None:
            self.texture_teacher.provenance=checkpoint.get('teacher_metadata',{'trained':False})


def make_model(cfg,num_class,camera_num,view_num,initialize=True):
    return MLDCReID(cfg,num_class,camera_num,view_num,initialize=initialize)
