"""MLDC+CNL baseline with optional training-only BatchFormer."""
import torch
from torch.nn import functional as F
from yacs.config import CfgNode
from .base_model import MLDCReID as Baseline,add_mldc_config as add_baseline_config
from .base_model import configure_head_learning_rates as configure_baseline_rates
from .prototype_attention import PrototypeAttentionHead,attention_distillation
from .pem import SparseTokenInteraction
from .batchformer import BatchFormer
from ..multiview_cross import cross_defaults, activation_factor, support_groups, cross_terms


def pem_defaults():
    return CfgNode(dict(ENABLED=False,LAYERS=[9,10,11,12],THRESHOLD=.5,
                       TEMPERATURE=1.,GATE_INIT=0.))


def add_mldc_config(cfg):
    cfg=add_baseline_config(cfg)
    cfg.PEM=pem_defaults()
    cfg.MVI_CROSS=cross_defaults()
    cfg.BATCHFORMER=CfgNode(dict(ENABLED=False,HEADS=4,DROPOUT=.5))
    cfg.NFC=CfgNode(dict(ENABLED=False,TEACHER_CHECKPOINT='',TEMPERATURE=4.,
        WEIGHT=1.,TARGET='nfc',EXTRACT_BATCH_SIZE=64,KNN_BLOCK_SIZE=256,
        FREEZE_ID_CLASSIFIER=True,EVAL_INITIAL=True,STUDENT_INIT='baseline',
        STUDENT_PROBABILITY='fixed_teacher_projection'))
    cfg.NFC_CENTER=CfgNode(dict(ENABLED=False,LOSS='contrastive',TEMPERATURE=.05,WEIGHT=.1,
        WARMUP_EPOCHS=20,RAMP_EPOCHS=5))
    cfg.PAD=CfgNode(dict(ENABLED=False,DIM=256,HEADS=8,LAYERS=1,
        FEATURE_LAYERS=[9,10,11,12],KV_LAYERS=[9,10,11,12],PARTS=1,
        USE_PROTOTYPE_QUERY=True,PROTO_WEIGHT=1.,ATT_WEIGHT=5.,PROTO_TEMPERATURE=.05,
        MOMENTUM=.5,MIN_IMAGES=2,WARMUP_EPOCHS=5,RAMP_EPOCHS=0,LR_FACTOR=5.,
        UPDATE_MODE='adaptive',AMU_TEMPERATURE=.05,EMBED_NORM='batchnorm',
        PROTO_DIM=768,STUDENT_FEATURE='refined_cls_mldc'))
    return cfg


def configure_head_learning_rates(cfg,model,optimizer):
    configure_baseline_rates(cfg,model,optimizer)
    ids={id(p) for n,p in model.named_parameters() if p.requires_grad and n.startswith('pad_head.')}
    for group in optimizer.param_groups:
        group_ids={id(p) for p in group['params']}
        if group_ids&ids:
            if not group_ids<=ids:
                raise ValueError('PAD and baseline parameters require separate optimizer groups')
            group['lr']*=cfg.PAD.LR_FACTOR


class MLDCReID(Baseline):
    descriptor_dim=768

    def __init__(self,cfg,num_classes,camera_num,view_num,initialize=True):
        super().__init__(cfg,num_classes,camera_num,view_num,
                         initialize=initialize and cfg.MODEL.PRETRAIN_CHOICE=='imagenet')
        self.pad_enabled=bool(cfg.PAD.ENABLED);self.pad_settings=cfg.PAD.clone()
        self.nfc_enabled=bool(cfg.NFC.ENABLED)
        self.nfc_center_enabled=bool(cfg.NFC_CENTER.ENABLED)
        self.nfc_center_mode=str(cfg.NFC_CENTER.LOSS)
        if self.nfc_center_mode not in ('contrastive','mean_shift'):
            raise ValueError('NFC_CENTER.LOSS must be contrastive or mean_shift')
        self.pem_settings=(cfg.PEM if 'PEM' in cfg else pem_defaults()).clone()
        self.pem_enabled=bool(self.pem_settings.ENABLED)
        self.batchformer_enabled=bool(cfg.BATCHFORMER.ENABLED)
        self.batchformer_settings=cfg.BATCHFORMER.clone()
        self.mvi_cross_settings=(cfg.MVI_CROSS if 'MVI_CROSS' in cfg else cross_defaults()).clone()
        self.mvi_cross_enabled=bool(self.mvi_cross_settings.ENABLED)
        self.mvi_cross_id_weight=float(cfg.MODEL.ID_LOSS_WEIGHT)
        if self.mvi_cross_enabled:
            m=self.mvi_cross_settings
            if (self.pad_enabled or self.pem_enabled or self.nfc_enabled or self.nfc_center_enabled
                    or self.batchformer_enabled):
                raise ValueError('MVI_CROSS requires plain MLDC+CNL; other auxiliary methods must be disabled')
            if (m.MIN_SUPPORTS!=2 or m.MAX_SUPPORTS!=3 or m.FUSION not in ('cross_attention','mean')
                    or m.WARMUP_EPOCHS<0 or m.RAMP_EPOCHS<1 or m.GROUP_ID_WEIGHT<0 or m.IP_WEIGHT<0
                    or self.readout!='mldc' or self.train_feature!='mldc' or self.retrieval!='mldc'
                    or self.neck_feat!='after' or self.mldc.selected_layers!=(9,10,11,12)
                    or cfg.MODEL.IF_LABELSMOOTH!='off'):
                raise ValueError('Require 2-3 supports, original CLS9-12 MLDC and unsmoothed ID')
        self.batchformer=None
        if self.batchformer_enabled:
            if self.pad_enabled or self.pem_enabled or self.nfc_enabled or self.nfc_center_enabled:
                raise ValueError('BatchFormer uses the plain MLDC+CNL baseline')
            if (self.readout!='mldc' or self.train_feature!='mldc' or self.retrieval!='mldc'
                    or self.neck_feat!='after' or self.mldc.selected_layers!=(9,10,11,12)):
                raise ValueError('BatchFormer requires the original CLS9-12 MLDC readout and BN neck')
            with torch.random.fork_rng(devices=[]):
                self.batchformer=BatchFormer(768,cfg.BATCHFORMER.HEADS,cfg.BATCHFORMER.DROPOUT)
        if self.nfc_center_enabled and (self.nfc_enabled or self.pad_enabled or self.pem_enabled):
            raise ValueError('Batch NFC centers require original MLDC; class KL/PAD/PEM must be disabled')
        if self.nfc_enabled and (self.pad_enabled or self.pem_enabled):
            raise ValueError('NFC class KD uses original MLDC with PAD/PEM disabled')
        self.cnl_threshold_reference=str(cfg.SOLVER.get('CNL_THRESHOLD_REFERENCE','all_nonself'))
        if self.pem_enabled and self.pad_enabled:
            raise ValueError('The minimal PEM experiment disables PAD and all prototype losses')
        s=self.pad_settings;self.feature_layers=tuple(s.FEATURE_LAYERS);self.pad_head=None
        if (self.feature_layers!=(9,10,11,12) or tuple(s.KV_LAYERS)!=self.feature_layers
                or s.PARTS!=1 or s.LAYERS!=1 or s.DIM<8 or s.HEADS<1 or s.DIM%s.HEADS
                or s.PROTO_TEMPERATURE<=0 or s.WARMUP_EPOCHS<0
                or s.RAMP_EPOCHS<0 or s.ATT_WEIGHT<0 or s.PROTO_WEIGHT<0 or s.LR_FACTOR<=0
                or s.UPDATE_MODE not in ['adaptive','fixed'] or s.AMU_TEMPERATURE<=0
                or s.EMBED_NORM!='batchnorm' or s.PROTO_DIM!=self.descriptor_dim
                or s.STUDENT_FEATURE!='refined_cls_mldc'):
            raise ValueError('Require matched P9-12, four refined CLS, one 768-d prototype and final descriptor')
        if self.pad_enabled:
            if (list(cfg.INPUT.SIZE_TRAIN)!=[256,128] or list(cfg.INPUT.SIZE_TEST)!=[256,128]
                    or list(cfg.MODEL.STRIDE_SIZE)!=[16,16] or not cfg.MODEL.STEM_CONV
                    or cfg.MODEL.SIE_CAMERA or cfg.MODEL.SIE_VIEW or self.neck_feat!='after'
                    or self.train_feature!='mldc' or self.retrieval!='mldc'
                    or self.mldc.selected_layers!=self.feature_layers):
                raise ValueError('Preserve the matched NightReID MLDC/CNL protocol')
            if not s.USE_PROTOTYPE_QUERY and s.ATT_WEIGHT:
                raise ValueError('Head-only control must have zero attention distillation weight')
            with torch.random.fork_rng(devices=[]):
                self.pad_head=PrototypeAttentionHead(768,s.DIM,s.HEADS,s.LAYERS,self.feature_layers)
        if self.pem_enabled:
            p=self.pem_settings
            if (tuple(p.LAYERS)!=(9,10,11,12) or self.train_feature!='mldc'
                    or self.retrieval!='mldc' or self.neck_feat!='after'
                    or self.mldc.selected_layers!=(9,10,11,12)):
                raise ValueError('Minimal PEM preserves the original CLS9-12 MLDC/CNL readout')
            for level in p.LAYERS:
                self.base.blocks[level-1].pem=SparseTokenInteraction(
                    768,p.THRESHOLD,p.TEMPERATURE,p.GATE_INIT)
        if initialize and cfg.MODEL.PRETRAIN_CHOICE=='self':
            self.load_param(cfg.MODEL.PRETRAIN_PATH)

    def pem_diagnostics(self):
        if not self.pem_enabled:return {}
        return {level:dict(gate=float(self.base.blocks[level-1].pem.gate.detach()),
                **{k:float(v) for k,v in self.base.blocks[level-1].pem.last_statistics.items()})
                for level in self.pem_settings.LAYERS}

    def _prototype_losses(self,unit,raw_layers,student,labels,indices,memory,epoch):
        zero=unit.sum()*0.
        stats=dict(anchors=len(unit),teacher_anchors=0)
        if student is not None:
            ent=-(student['attention'].detach()*student['log_attention'].detach()).sum(-1)
            stats['student_attention_entropy']=float(ent.mean())
            for i,level in enumerate(self.feature_layers):
                stats['attention_kl_layer%d'%level]=0.
                stats['attention_entropy_layer%d'%level]=float(ent[:,i].mean())
        terms=dict(attention=zero,prototype=zero,student_id=zero,teacher_id=zero,
                   student_features=unit,teacher_features=None,
                   update_features=unit.detach() if self.pad_enabled else None,stats=stats)
        if not self.pad_enabled or not self.training or memory is None:
            return terms,None
        if labels is None or indices is None or epoch is None:
            raise ValueError('Training PAD needs identity labels, stable indices and epoch')
        memory.validate(unit,labels,indices);s=self.pad_settings
        with torch.autocast(device_type=unit.device.type,enabled=False):
            keys,columns=memory.classification_keys(unit,labels,indices)
            student_id=F.cross_entropy(unit.float()@keys.T/s.PROTO_TEMPERATURE,columns)
        prototype,active=memory.lookup(labels)
        active=active&(int(epoch)>s.WARMUP_EPOCHS)&bool(s.USE_PROTOTYPE_QUERY)
        teacher=None;teacher_id=zero;attention=zero;updates=unit.detach().clone()
        if active.any():
            spatial={level:{name:value[active] for name,value in student['spatial'][level].items()}
                     for level in self.feature_layers}
            teacher=self.pad_head.teacher(prototype[active],raw_layers[:,active],spatial)
            teacher_mixed=self.mldc(teacher['refined_layers'])
            teacher_neck=self.bottleneck(teacher_mixed)
            teacher_unit=F.normalize(teacher_neck.float(),dim=1)
            teacher.update(mixed=teacher_mixed,embedding=teacher_neck,unit=teacher_unit)
            with torch.autocast(device_type=unit.device.type,enabled=False):
                teacher_id=F.cross_entropy(teacher_unit.float()@keys.T/s.PROTO_TEMPERATURE,columns[active])
                per=F.kl_div(student['log_attention'][active].float(),
                             teacher['attention'].detach().float(),reduction='none').sum(-1)
            attention=attention_distillation(student['log_attention'][active],teacher['attention'])
            for i,level in enumerate(self.feature_layers):
                stats['attention_kl_layer%d'%level]=float(per[:,i].detach().mean())
            updates[active]=teacher_unit.detach()
        terms.update(attention=attention,prototype=.5*(student_id+teacher_id) if teacher is not None else student_id,
                     student_id=student_id,teacher_id=teacher_id,
                     teacher_features=teacher['unit'] if teacher is not None else None,update_features=updates)
        stats.update(teacher_anchors=int(active.sum()),prototype_classes=int(memory.valid.sum()),
                     identity_dimension=self.descriptor_dim)
        return terms,teacher

    def forward(self,x,label=None,cam_label=None,view_label=None,return_components=False,
                return_cnl_features=False,return_distillation=False,sample_indices=None,
                prototype_memory=None,epoch=None,return_pad_details=False):
        if not self.training and (prototype_memory is not None or return_distillation):
            raise ValueError('Evaluation must not access training identity prototypes')
        if self.training and return_distillation and not return_cnl_features:
            raise ValueError('Return the final descriptor shared with CNL')
        if self.training and return_distillation and self.pad_enabled and prototype_memory is None:
            raise ValueError('PAD losses need training prototype memory')
        cross_active=False
        if self.training and self.mvi_cross_enabled:
            if not return_distillation or label is None or cam_label is None or sample_indices is None or epoch is None:
                raise ValueError('MVI_CROSS needs indexed labeled training batches and loss outputs')
            factor=activation_factor(epoch,self.mvi_cross_settings)
            cross_active=factor>0.
            grouping=support_groups(label,cam_label,sample_indices,epoch,self.mvi_cross_settings) if cross_active else (
                label.new_empty(0),label.new_empty((0,2)),torch.empty((0,2),dtype=torch.bool,device=x.device),
                label.new_full((len(label),),-1),dict(unique_images=0,eligible_ids=0,groups=0,active_anchors=0,
                    support_images=0,crosscam_groups=0))
        if cross_active and len(grouping[0]):
            raw_layers,patches=self.base(x,cam_label=cam_label,view_label=view_label,
                                        return_patch_layers=(8,9,10,11))
            student=None;refined=raw_layers
        elif self.pad_enabled:
            raw_layers,patches=self.base(x,cam_label=cam_label,view_label=view_label,
                                        return_patch_layers=self.feature_layers)
            student=self.pad_head.student(raw_layers,patches)
            refined=student['refined_layers']
        else:
            raw_layers=self.base(x,cam_label=cam_label,view_label=view_label)
            patches={};student=None;refined=raw_layers
        mixed=self.mldc(refined)
        # Both paths share one BN call and the same classifier. Only the
        # original N rows are exposed to Triplet, CNL and memory updates.
        bf_neck=None
        if self.training and self.batchformer_enabled:
            if not return_distillation:
                raise ValueError('BatchFormer training requires both classification outputs')
            joined=torch.cat((mixed,self.batchformer(mixed)),dim=0)
            neck,bf_neck=self.bottleneck(joined).chunk(2,dim=0)
        else:
            neck=self.bottleneck(mixed)
        unit=F.normalize(neck.float(),dim=1)
        if student is not None:student.update(unit=unit,embedding=neck)
        terms,teacher=self._prototype_losses(unit,raw_layers,student,label,sample_indices,
                                             prototype_memory,epoch)
        raw_global=self.mldc(raw_layers) if self.pad_enabled and (self.training or return_pad_details) else mixed
        if self.pad_enabled and (self.training or return_pad_details):
            terms['stats']['residual_norm_ratio']=float(
                (mixed-raw_global).detach().float().norm(dim=1).mean()/
                raw_global.detach().float().norm(dim=1).mean().clamp_min(1e-12))
        if return_pad_details:
            return dict(retrieval=neck,mixed=mixed,global_feature=raw_global,layers=raw_layers,
                        refined_layers=refined,patches=patches,student=student,student_features=unit,
                        teacher=teacher,terms=terms)
        if self.training:
            if bf_neck is not None:
                scores,bf_scores=self.classifier(self.dropout(torch.cat((neck,bf_neck),dim=0))).chunk(2,dim=0)
                terms['batchformer_scores']=bf_scores
                result=(scores,mixed)
            else:
                result=(self.classifier(self.dropout(neck)),mixed)
            if self.mvi_cross_enabled:
                terms['mvi_cross']=cross_terms(self,raw_layers,patches,grouping,mixed,unit,label,factor)
            if return_cnl_features:result=(*result,unit)
            if return_distillation:result=(*result,terms)
            return result
        return (neck,mixed,refined) if return_components else neck

    def distillation_loss(self,terms,epoch):
        if not self.pad_enabled:return terms['attention']*0.
        return self.pad_settings.PROTO_WEIGHT*terms['prototype']+self.attention_weight(epoch)*terms['attention']

    def attention_weight(self,epoch):
        s=self.pad_settings
        if not self.pad_enabled or int(epoch)<=s.WARMUP_EPOCHS:return 0.
        ramp=1. if s.RAMP_EPOCHS==0 else min(1.,float(epoch-s.WARMUP_EPOCHS)/s.RAMP_EPOCHS)
        return float(s.ATT_WEIGHT)*ramp

    def architecture(self):
        if self.mvi_cross_enabled:
            return dict(variant='mldc_cnl_support_only_cross_attention_20261101',descriptor_dim=768,
                supports='2-3 distinct same-ID images; anchor and all its augmentation copies excluded',
                anchor='one rotating anchor source image per eligible identity per batch',
                settings=dict(self.mvi_cross_settings),new_parameters=0,
                fusion='support mean CLS8 queries support patches8-11 with original blocks9-12 projections/LN/FFN',
                readout='shared CLS9-12 MLDC, BN affine and classifier',
                bn='only single images update statistics; groups use detached current single batch moments',
                ip='normalized cosine; support target detached only in IP',ema=False,
                threshold_reference=self.cnl_threshold_reference,
                inference='original independent single-image backbone+MLDC+BN; no support images')
        if self.batchformer_enabled:
            return dict(variant='mldc_cnl_batchformer_20261101',descriptor_dim=768,
                placement='MLDC output, before shared BNNeck and classifier',
                heads=self.batchformer_settings.HEADS,dropout=self.batchformer_settings.DROPOUT,
                encoder_layers=1,feedforward_dim=768,activation='relu',norm_first=False,
                attention='batch dimension [N,1,768]; no positional encoding or label mask',
                identity='0.5*(original CE + BatchFormer CE), shared classifier',
                metric='original-feature Triplet and 0.1 CNL; original rows only in CNL memory',
                bn='one shared BN call on concatenated original and transformed rows during training',
                threshold_reference=self.cnl_threshold_reference,
                inference='original independent single-image MLDC+BN; BatchFormer bypassed')
        if self.nfc_center_enabled:
            if self.nfc_center_mode=='mean_shift':
                return dict(variant='mldc_cnl_batch_mean_shift_20261031',descriptor_dim=768,
                    pad_enabled=False,pem_enabled=False,threshold_reference=self.cnl_threshold_reference,
                    loss='ID+Triplet+weighted CNL+batch reciprocal-neighbor mean shift',
                    neighborhood='own unlabeled reciprocal top2, unique image indices, self copies excluded',
                    target='detached arithmetic neighbor mean, NOT renormalized; no extra negatives or KL',
                    inference='original single-image MLDC plus BN768; no neighbors or extra heads',new_parameters=0)
            return dict(variant='mldc_cnl_batch_nfc_center_20261031',descriptor_dim=768,
                pad_enabled=False,pem_enabled=False,threshold_reference=self.cnl_threshold_reference,
                loss='ID+Triplet+weighted CNL+batch NFC center contrast',
                neighborhood='current batch, unlabeled reciprocal top2, duplicate indices coalesced',
                target='detached leave-anchor-image-out local centers with at least two images',
                inference='original single-image MLDC plus BN768; no neighbors or extra heads',new_parameters=0)
        if self.nfc_enabled:
            return dict(variant='mldc_cnl_nfc_class_kl_20261031',descriptor_dim=768,
                pad_enabled=False,pem_enabled=False,threshold_reference=self.cnl_threshold_reference,
                loss='ID+Triplet+weighted CNL+NFC identity probability KL',
                teacher='fixed unlabeled train-only NFC and original classifier',
                inference='original single-image MLDC plus BN768; teacher absent',new_parameters=0)
        if not self.pad_enabled and not self.pem_enabled:
            return dict(variant='mldc_cnl_threshold_reference_20261030',descriptor_dim=768,
                pad_enabled=False,pem_enabled=False,
                threshold_reference=self.cnl_threshold_reference,
                fusion='original learned MLDC over unmodified post-FFN CLS9-12',
                loss='original ID + Triplet + 0.1 cross-camera CNL',
                inference='original MLDC plus BN768; no extra branches or identity memory',
                new_parameters=0)
        if self.pem_enabled:
            return dict(variant='mldc_cnl_pem_minimal_20261030',descriptor_dim=768,
                pad_enabled=False,pem_enabled=True,
                settings={k:self.pem_settings[k] for k in ['LAYERS','THRESHOLD','TEMPERATURE','GATE_INIT']},
                query='existing per-image CLS; all CLS and patch tokens jointly updated',
                placement='after MHSA residual and before the original FFN in blocks9-12',
                affinity='parameter-free layernorm then L2 cosine; threshold before row softmax; retain diagonal',
                update='T + gate * softmax(masked affinity / temperature) @ T',
                fusion='original learned MLDC applied to the four post-FFN CLS snapshots',
                loss='original ID + Triplet + weighted cross-camera CNL only',
                prototype='no identity prototype memory and no prototype query',
                inference='same CLS/spatial interaction as training; no identity labels or identity memory',
                new_parameters='one zero-initialized scalar gate per block; four scalars total')
        return dict(variant='mldc_pad_multilevel_cls_20261029',descriptor_dim=768,pad_enabled=self.pad_enabled,
            settings={k:self.pad_settings[k] for k in ['DIM','HEADS','LAYERS','FEATURE_LAYERS',
                                                       'KV_LAYERS','PARTS','PROTO_DIM','STUDENT_FEATURE']},
            student_query='raw CLS_l reads post-Block-l patches, independently for l=9,10,11,12',
            fusion='original learned MLDC weights applied after four CLS refinements',
            teacher_query='same single 768-d identity prototype at all four levels',
            teacher_feature='same MLDC aggregation of four prototype-refined CLS; same final BN768',
            parameter_sharing='four independent level modules; teacher/student share Q/K/V/output within each level',
            distillation='spatial KL per backbone level and head; teacher target detached; mean over batch/level/head',
            prototype='one final 768-d table; teacher aggregated embedding updates it with AMU',
            attention='one direct cross-attention per level, no MHCA stack, no FFN, no prototype-query residual',
            inference='four CLS refiners and original MLDC retained; no identity prototypes',
            embedding_normalization='one final shared BN768 then L2; no per-level identity embedding')

    def inference_state_dict(self):return self.state_dict()

    def load_param(self,path):
        cp=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        if cp.get('architecture')!=self.architecture():
            raise ValueError('Use a matching multilevel CLS checkpoint; older architectures need their source backup')
        self.load_state_dict(cp['state_dict'],strict=True);self.checkpoint_epoch=int(cp.get('epoch',0))

    def load_baseline(self,path):
        cp=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        missing,unexpected=self.load_state_dict(cp.get('state_dict',cp),strict=False)
        gate_names={f'base.blocks.{level-1}.pem.gate' for level in self.pem_settings.LAYERS} if self.pem_enabled else set()
        if unexpected or any(not k.startswith(('pad_head.','batchformer.')) and k not in gate_names for k in missing):
            raise ValueError('Original baseline backbone/MLDC/BN do not match')
        if self.pem_enabled and any(float(self.base.blocks[level-1].pem.gate.detach())!=0.
                                    for level in self.pem_settings.LAYERS):
            raise ValueError('Baseline equivalence requires zero-initialized PEM gates')
        if self.pad_enabled and any(torch.count_nonzero(h.output_projection.weight)
                                   for h in self.pad_head.level_heads.values()):
            raise ValueError('Baseline equivalence requires zero-initialized CLS residual projections')


def make_model(cfg,num_class,camera_num,view_num,initialize=True):
    return MLDCReID(cfg,num_class,camera_num,view_num,initialize=initialize)
