"""Unmodified pretrained ViT plus a frozen, training-only independent teacher."""
import torch
from yacs.config import CfgNode
from .base_model import MLDCReID as Baseline,add_mldc_config as add_baseline_config
from .base_model import configure_head_learning_rates as configure_baseline_rates
from .texture_teacher import TextureTeacher
from .final_relation import local_pair_scores,relative_distillation
from .final_contrast import FinalContrast


def add_mldc_config(cfg):
    cfg=add_baseline_config(cfg)
    cfg.TEXTURE=CfgNode(dict(ENABLED=True,BANK='texture',GRID=[8,4],DIM=128,
        TEACHER_INIT_PATH='',TEACHER_PATH='',TEACHER_EPOCHS=10,TEACHER_LR=.0004,
        TEACHER_HEAD_LR_FACTOR=5.,TEACHER_WARMUP_EPOCHS=2,TEACHER_STRUCTURE_WEIGHT=.1,TEACHER_ILLUM_WEIGHT=.1,
        WEIGHT=.1,RAMP_EPOCHS=20,TEMPERATURE=.2,GAP=.05,HARD_NEGATIVES=8,MATCH_TOPK=4,
        BAND_ROWS=2,MIN_PATCH_CONF=.5,MIN_PAIR_CONF=.25,MIN_MATCHES=2,RELIABILITY_EXPOSURE=.7,
        OBJECTIVE='legacy_relation',CONTRAST_DIM=128,PROJECTION_LR_FACTOR=5.,CROSS_CAMERA_POSITIVES=False))
    return cfg


def configure_head_learning_rates(cfg,model,optimizer):
    configure_baseline_rates(cfg,model,optimizer)
    projections={id(p) for name,p in model.named_parameters() if p.requires_grad and name.startswith('texture_contrast.')}
    for group in optimizer.param_groups:
        identifiers={id(p) for p in group['params']}
        if identifiers&projections:
            if not identifiers<=projections: raise ValueError('Mixed projection/backbone optimizer group')
            group['lr']*=cfg.TEXTURE.PROJECTION_LR_FACTOR


class MLDCReID(Baseline):
    def __init__(self,cfg,num_classes,camera_num,view_num,initialize=True,teacher_override=None):
        super().__init__(cfg,num_classes,camera_num,view_num,
            initialize=initialize and cfg.MODEL.PRETRAIN_CHOICE=='imagenet')
        self.texture_enabled=bool(cfg.TEXTURE.ENABLED); self.texture_settings=cfg.TEXTURE.clone()
        self.texture_teacher=None; self.texture_contrast=None
        if cfg.TEXTURE.OBJECTIVE not in ('legacy_relation','identity_infonce'):
            raise ValueError('Unknown final-feature distillation objective')
        if self.texture_enabled:
            if (list(cfg.INPUT.SIZE_TRAIN)!=[256,128] or list(cfg.MODEL.STRIDE_SIZE)!=[16,16]
                    or list(cfg.INPUT.SIZE_TEST)!=[256,128]
                    or list(cfg.INPUT.PIXEL_MEAN)!=[.5,.5,.5] or list(cfg.INPUT.PIXEL_STD)!=[.5,.5,.5]
                    or not cfg.MODEL.STEM_CONV or cfg.TEST.NECK_FEAT!='after'
                    or cfg.MODEL.MLDC_TRAIN_FEATURE!='mldc' or cfg.TEST.MLDC_RETRIEVAL!='mldc'):
                raise ValueError('Distill the original matched BN-after MLDC retrieval descriptor')
            with torch.random.fork_rng(devices=[]):
                self.texture_teacher=teacher_override or TextureTeacher(num_classes,cfg.TEXTURE.BANK,
                    cfg.TEXTURE.GRID,cfg.TEXTURE.DIM)
                if cfg.TEXTURE.OBJECTIVE=='identity_infonce':
                    self.texture_contrast=FinalContrast(cfg.TEXTURE.CONTRAST_DIM,cfg.TEXTURE.TEMPERATURE,
                        cfg.TEXTURE.CROSS_CAMERA_POSITIVES)
            if initialize and cfg.MODEL.PRETRAIN_CHOICE=='imagenet' and teacher_override is None:
                self.texture_teacher.load_trained(cfg.TEXTURE.TEACHER_PATH,cfg.TEXTURE.TEACHER_EPOCHS)
            self.texture_teacher.requires_grad_(False).eval()
        if initialize and cfg.MODEL.PRETRAIN_CHOICE=='self': self.load_param(cfg.MODEL.PRETRAIN_PATH)

    def train(self,mode=True):
        super().train(mode)
        if self.texture_teacher is not None: self.texture_teacher.eval()
        return self

    def teacher_targets(self,images):
        s=self.texture_settings
        result=self.texture_teacher.describe(images.detach().float(),s.RELIABILITY_EXPOSURE)
        return local_pair_scores(result['tokens'],result['confidence'],s.GRID,s.MATCH_TOPK,
            s.BAND_ROWS,s.MIN_PATCH_CONF,s.MIN_MATCHES)

    def forward(self,x,label=None,cam_label=None,view_label=None,return_components=False,
                return_cnl_features=False,return_distillation=False,sample_indices=None):
        result=super().forward(x,label,cam_label,view_label,return_components,return_cnl_features)
        if not self.training or not return_distillation: return result
        if not return_cnl_features: raise ValueError('Return the original final CNL descriptors for distillation')
        if not self.texture_enabled: return (*result,dict(relation=result[1].sum()*0,stats={}))
        if label is None or cam_label is None or sample_indices is None:
            raise ValueError('Use training identity, camera and stable image indices for relative distillation')
        s=self.texture_settings
        with torch.cuda.amp.autocast(enabled=False):
            if s.OBJECTIVE=='identity_infonce':
                with torch.no_grad(): teacher=self.texture_teacher(x.detach().float())['unit']
                loss,stats=self.texture_contrast(result[2],teacher,label,cam_label,sample_indices)
            else:
                with torch.no_grad(): score,confidence=self.teacher_targets(x)
                loss,stats=relative_distillation(result[2],score,confidence,label,cam_label,sample_indices,
                    s.TEMPERATURE,s.GAP,s.HARD_NEGATIVES,s.MIN_PAIR_CONF)
        return (*result,dict(relation=loss,stats=stats))

    def distillation_loss(self,terms,epoch):
        if not self.texture_enabled: return terms['relation']*0
        s=self.texture_settings; ramp=min(1.,max(0.,float(epoch)/max(int(s.RAMP_EPOCHS),1)))
        return float(s.WEIGHT)*ramp*terms['relation']

    def architecture(self):
        settings=self.texture_settings
        keys=['BANK','GRID','DIM','TEMPERATURE','GAP','HARD_NEGATIVES','MATCH_TOPK','BAND_ROWS',
            'MIN_PATCH_CONF','MIN_PAIR_CONF','MIN_MATCHES','RELIABILITY_EXPOSURE','WEIGHT','RAMP_EPOCHS']
        keys+=['OBJECTIVE','CONTRAST_DIM','PROJECTION_LR_FACTOR','CROSS_CAMERA_POSITIVES']
        return dict(variant='convstream_final_texture_20261026',descriptor_dim=768,
            texture_enabled=self.texture_enabled,distillation_target='L2-normalized final BN-after MLDC descriptor',
            teacher='independent DCM ConvStream 48/96/192',settings={k:settings[k] for k in keys},
            inference='original ViT + MLDC + BNneck only')

    def inference_state_dict(self):
        return {k:v for k,v in self.state_dict().items() if not k.startswith(('texture_teacher.','texture_contrast.'))}

    def load_param(self,path):
        checkpoint=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        if checkpoint.get('architecture')!=self.architecture(): raise ValueError('Checkpoint architecture differs')
        self.load_state_dict(checkpoint['state_dict'],strict=True); self.checkpoint_epoch=int(checkpoint['epoch'])
        if self.texture_teacher is not None:
            self.texture_teacher.provenance=checkpoint.get('teacher_metadata') or {'trained':False}


def make_model(cfg,num_class,camera_num,view_num,initialize=True):
    return MLDCReID(cfg,num_class,camera_num,view_num,initialize=initialize)
