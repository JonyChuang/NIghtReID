"""One fused student descriptor for retrieval, identities, prototypes and PAD."""
import torch
from torch.nn import functional as F
from yacs.config import CfgNode
from .base_model import MLDCReID as Baseline, add_mldc_config as add_baseline_config
from .base_model import configure_head_learning_rates as configure_baseline_rates
from .prototype_attention import PrototypeAttentionHead, attention_distillation


def add_mldc_config(cfg):
    cfg = add_baseline_config(cfg)
    cfg.PAD = CfgNode(dict(ENABLED=True, DIM=256, HEADS=8, LAYERS=3, FFN_EXPANSION=4,
        KV_LAYER=12, PARTS=1, USE_PROTOTYPE_QUERY=True, PROTO_WEIGHT=1., ATT_WEIGHT=5.,
        PROTO_TEMPERATURE=.05, MOMENTUM=.5, MIN_IMAGES=2, WARMUP_EPOCHS=5,
        RAMP_EPOCHS=0, LR_FACTOR=5., UPDATE_MODE='adaptive', AMU_TEMPERATURE=.05,
        EMBED_NORM='batchnorm', PROTO_DIM=768, STUDENT_FEATURE='fused'))
    return cfg


def configure_head_learning_rates(cfg, model, optimizer):
    configure_baseline_rates(cfg, model, optimizer)
    ids = {id(p) for n,p in model.named_parameters() if p.requires_grad and n.startswith('pad_head.')}
    for group in optimizer.param_groups:
        group_ids = {id(p) for p in group['params']}
        if group_ids & ids:
            if not group_ids <= ids:
                raise ValueError('PAD and baseline parameters require separate optimizer groups')
            group['lr'] *= cfg.PAD.LR_FACTOR


class MLDCReID(Baseline):
    descriptor_dim = 768

    def __init__(self, cfg, num_classes, camera_num, view_num, initialize=True):
        super().__init__(cfg, num_classes, camera_num, view_num,
                         initialize=initialize and cfg.MODEL.PRETRAIN_CHOICE=='imagenet')
        self.pad_enabled = bool(cfg.PAD.ENABLED)
        self.pad_settings = cfg.PAD.clone()
        self.pad_head = None
        s = self.pad_settings
        if (s.KV_LAYER != 12 or s.PARTS != 1 or s.LAYERS != 3 or s.DIM < 8
                or s.HEADS < 1 or s.DIM % s.HEADS or s.FFN_EXPANSION < 1
                or s.PROTO_TEMPERATURE <= 0 or s.WARMUP_EPOCHS < 0 or s.RAMP_EPOCHS < 0
                or s.ATT_WEIGHT < 0 or s.PROTO_WEIGHT < 0 or s.LR_FACTOR <= 0
                or s.UPDATE_MODE not in ['adaptive','fixed'] or s.AMU_TEMPERATURE <= 0
                or s.EMBED_NORM != 'batchnorm' or s.PROTO_DIM != self.descriptor_dim
                or s.STUDENT_FEATURE != 'fused'):
            raise ValueError('PAD requires a single final fused 768-d student identity space')
        if self.pad_enabled:
            if (list(cfg.INPUT.SIZE_TRAIN)!=[256,128] or list(cfg.INPUT.SIZE_TEST)!=[256,128]
                    or list(cfg.MODEL.STRIDE_SIZE)!=[16,16] or not cfg.MODEL.STEM_CONV
                    or cfg.MODEL.SIE_CAMERA or cfg.MODEL.SIE_VIEW
                    or self.neck_feat!='after' or self.train_feature!='mldc'
                    or self.retrieval!='mldc' or self.mldc.selected_layers!=(9,10,11,12)):
                raise ValueError('Preserve the original NightReID MLDC/CNL descriptor protocol')
            if not s.USE_PROTOTYPE_QUERY and s.ATT_WEIGHT:
                raise ValueError('A head-only control must set PAD.ATT_WEIGHT to zero')
            with torch.random.fork_rng(devices=[]):
                self.pad_head = PrototypeAttentionHead(768,s.DIM,s.HEADS,s.LAYERS,s.FFN_EXPANSION)
        if initialize and cfg.MODEL.PRETRAIN_CHOICE=='self':
            self.load_param(cfg.MODEL.PRETRAIN_PATH)

    def _prototype_losses(self, unit, global_feature, spatial, student,
                          labels=None, indices=None, memory=None, epoch=None):
        zero = unit.sum()*0.
        terms = dict(attention=zero,prototype=zero,student_id=zero,teacher_id=zero,
                     student_features=unit,teacher_features=None,
                     update_features=unit.detach() if self.pad_enabled else None,
                     stats=dict(anchors=len(unit),teacher_anchors=0))
        teacher = None
        if not self.pad_enabled or not self.training or memory is None:
            return terms,teacher
        if labels is None or indices is None or epoch is None:
            raise ValueError('PAD training needs labels, stable image indices and epoch')
        memory.validate(unit,labels,indices)
        s = self.pad_settings
        with torch.autocast(device_type=unit.device.type, enabled=False):
            keys,target_columns = memory.classification_keys(unit,labels,indices)
            student_id = F.cross_entropy(unit.float() @ keys.T/s.PROTO_TEMPERATURE,
                                         target_columns)
        proto,active = memory.lookup(labels)
        active = active & (int(epoch)>s.WARMUP_EPOCHS) & bool(s.USE_PROTOTYPE_QUERY)
        update_features = unit.detach().clone()
        teacher_id,attention = zero,zero
        if active.any():
            teacher = self.pad_head.teacher(proto[active],spatial[active])
            # Both identities inhabit the SAME final fusion/BN space. The
            # prototype changes the teacher's spatial context on this image.
            teacher_mixed = global_feature[active] + self.pad_head.residual_projection(teacher['context'])
            teacher_neck = self.bottleneck(teacher_mixed)
            teacher_unit = F.normalize(teacher_neck.float(),dim=1)
            teacher.update(mixed=teacher_mixed,embedding=teacher_neck,unit=teacher_unit)
            with torch.autocast(device_type=unit.device.type, enabled=False):
                teacher_id = F.cross_entropy(teacher_unit.float() @ keys.T/s.PROTO_TEMPERATURE,
                                             target_columns[active])
            attention = attention_distillation(student['log_attention'][active],teacher['attention'])
            update_features[active] = teacher_unit.detach()
        prototype = .5*(student_id+teacher_id) if teacher is not None else student_id
        terms.update(attention=attention,prototype=prototype,student_id=student_id,
                     teacher_id=teacher_id,teacher_features=teacher['unit'] if teacher is not None else None,
                     update_features=update_features,
                     stats=dict(anchors=len(labels),teacher_anchors=int(active.sum()),
                         prototype_classes=int(memory.valid.sum()),identity_dimension=self.descriptor_dim,
                         student_attention_entropy=float((-(student['attention'].detach()*student['log_attention'].detach()).sum(-1)).mean())))
        return terms,teacher

    def forward(self,x,label=None,cam_label=None,view_label=None,return_components=False,
                return_cnl_features=False,return_distillation=False,sample_indices=None,
                prototype_memory=None,epoch=None,return_pad_details=False):
        if not self.training and (prototype_memory is not None or return_distillation):
            raise ValueError('Evaluation must not access training identity prototypes')
        if self.training and return_distillation and not return_cnl_features:
            raise ValueError('Return the single final descriptor shared with CNL')
        if self.training and return_distillation and self.pad_enabled and prototype_memory is None:
            raise ValueError('PAD losses require the official training prototype memory')
        layers,patches = self.base(x,cam_label=cam_label,view_label=view_label,return_final_patches=True)
        global_feature = self.mldc(layers)
        mixed,context,spatial = global_feature,None,None
        if self.pad_enabled:
            spatial = self.pad_head.prepare(patches)
            mixed,context = self.pad_head.fuse(global_feature,spatial)
        neck = self.bottleneck(mixed)
        unit = F.normalize(neck.float(),dim=1)
        student = None
        if self.pad_enabled and (self.training or return_pad_details):
            # This happens AFTER fusion and the final neck, with no detach.
            student = self.pad_head.student(unit,spatial)
            student['embedding'] = neck
        terms,teacher = self._prototype_losses(unit,global_feature,spatial,student,
                                               label,sample_indices,prototype_memory,epoch)
        if self.pad_enabled and (self.training or return_pad_details):
            terms['stats']['residual_norm_ratio'] = float(
                (mixed-global_feature).detach().float().norm(dim=1).mean()/
                global_feature.detach().float().norm(dim=1).mean().clamp_min(1e-12))
        if return_pad_details:
            return dict(retrieval=neck,mixed=mixed,global_feature=global_feature,layers=layers,
                        patches=patches,student=student,student_features=unit,
                        teacher=teacher,context=context,terms=terms)
        if self.training:
            result=(self.classifier(self.dropout(neck)),mixed)
            if return_cnl_features:
                result=(*result,unit)
            if return_distillation:
                result=(*result,terms)
            return result
        return (neck,mixed,layers) if return_components else neck

    def distillation_loss(self,terms,epoch):
        if not self.pad_enabled:
            return terms['attention']*0.
        return self.pad_settings.PROTO_WEIGHT*terms['prototype']+self.attention_weight(epoch)*terms['attention']

    def attention_weight(self,epoch):
        s=self.pad_settings
        if not self.pad_enabled or int(epoch)<=s.WARMUP_EPOCHS:
            return 0.
        ramp=1. if s.RAMP_EPOCHS==0 else min(1.,float(epoch-s.WARMUP_EPOCHS)/s.RAMP_EPOCHS)
        return float(s.ATT_WEIGHT)*ramp

    def architecture(self):
        return dict(variant='mldc_pad_fused_student_20261029',descriptor_dim=768,pad_enabled=self.pad_enabled,
            settings={k:self.pad_settings[k] for k in ['DIM','HEADS','LAYERS','FFN_EXPANSION','KV_LAYER','PARTS','PROTO_DIM','STUDENT_FEATURE']},
            fusion='g=MLDC(CLS9-12); f=g+zero-initialized context projection; z=normalize(BN768(f))',
            student_query='the exact final z used for retrieval, CNL and student prototype CE',
            teacher_query='768-d identity prototype projected to attention width',
            teacher_feature='same-image g plus prototype-conditioned context; same BN768 as student',
            spatial='post-Block12 P12 only, 128 positions; no ViT attention weight reuse',
            parameter_sharing='three independent-depth MHCA+FFN layers shared by fusion, student readout and teacher',
            embedding_normalization='one shared final BN768 and L2; no auxiliary 256-d identity embedding',
            inference='fusion construction retained; attention readout and prototypes absent from standard retrieval')

    def inference_state_dict(self):
        return self.state_dict()

    def load_param(self,path):
        checkpoint=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        if checkpoint.get('architecture')!=self.architecture():
            raise ValueError('Use a matching fused-student PAD checkpoint; old dual-embedding checkpoints need their backup')
        self.load_state_dict(checkpoint['state_dict'],strict=True)
        self.checkpoint_epoch=int(checkpoint.get('epoch',0))

    def load_baseline(self,path):
        checkpoint=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        state=checkpoint.get('state_dict',checkpoint)
        missing,unexpected=self.load_state_dict(state,strict=False)
        if unexpected or any(not k.startswith('pad_head.') for k in missing):
            raise ValueError('Baseline checkpoint does not match original backbone/MLDC/BN')
        if self.pad_enabled and torch.count_nonzero(self.pad_head.residual_projection.weight):
            raise ValueError('Baseline equivalence requires a fresh zero residual projection')


def make_model(cfg,num_class,camera_num,view_num,initialize=True):
    return MLDCReID(cfg,num_class,camera_num,view_num,initialize=initialize)
