import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID
from .backbones.swin_transformer import swin_base_patch4_window7_224, swin_small_patch4_window7_224
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a
import torch.nn.functional as F
import math

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        elif model_name == 'resnet50_ibn_a':
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride)
            print('using resnet50_ibn_a as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))


        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        if self.reduce_feat_dim:
            self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
            self.fcneck.apply(weights_init_xavier)
            self.in_planes = cfg.MODEL.FEAT_DIM

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)

        if pretrain_choice == 'self':
            self.load_param(model_path)


    def forward(self, x, label=None, **kwargs):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        if self.reduce_feat_dim:
            global_feat = self.fcneck(global_feat)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)
        if self.dropout_rate > 0:
            feat = self.dropout(feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            if 'classifier' in i:
                continue
            elif 'module' in i:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            else:
                self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    #  def load_param(self, trained_path):
        #  param_dict = torch.load(trained_path, map_location = 'cpu')
        #  for i in param_dict:
            #  try:
                #  self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            #  except:
                #  continue
        #  print('Loading pretrained model from {}'.format(trained_path))




class QNormContextCrossAttention(nn.Module):
    """Use CLS to query Q-Norm guided structural contexts."""

    def __init__(
            self,
            dim,
            context_tokens=4,
            num_heads=12,
            temperature=1.0,
            fusion_alpha=0.1,
            learnable_alpha=False,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                "dim must be divisible by num_heads."
            )

        if context_tokens < 2:
            raise ValueError(
                "context_tokens must be at least 2."
            )

        self.context_tokens = context_tokens
        self.temperature = float(temperature)
        initial_alpha = float(fusion_alpha)

        if not 0.0 <= initial_alpha <= 1.0:
            raise ValueError(
                "fusion_alpha must be between 0 and 1."
            )

        self.learnable_alpha = bool(learnable_alpha)

        if self.learnable_alpha:
            if not 0.0 < initial_alpha < 1.0:
                raise ValueError(
                    "Learnable alpha must start strictly between 0 and 1."
                )

            initial_logit = math.log(
                initial_alpha / (1.0 - initial_alpha)
            )

            self.fusion_alpha_logit = nn.Parameter(
                torch.tensor(
                    initial_logit,
                    dtype=torch.float32,
                )
            )
        else:
            self.register_buffer(
                "fusion_alpha_value",
                torch.tensor(
                    initial_alpha,
                    dtype=torch.float32,
                ),
            )

        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,
        )

        self.output_proj = nn.Linear(dim, dim)

        self.last_attention = None
        self.last_context_weights = None
        self.last_visualization = None

    def get_fusion_alpha(self):
        if self.learnable_alpha:
            return torch.sigmoid(
                self.fusion_alpha_logit
            )

        return self.fusion_alpha_value

    def forward(self, student_tokens, teacher_qnorm, capture_maps=False):
        cls_token = student_tokens[:, :1, :]
        patch_tokens = student_tokens[:, 1:, :]
        batch_size, num_patches, dim = patch_tokens.shape

        if teacher_qnorm.shape != (batch_size, num_patches):
            raise RuntimeError(
                f"Q-Norm shape {teacher_qnorm.shape} does not match "
                f"patch shape {(batch_size, num_patches)}."
            )

        if num_patches % self.context_tokens != 0:
            raise RuntimeError("Patch count must be divisible by context_tokens.")

        raw_qnorm = teacher_qnorm.float().detach()
        qnorm_mean = raw_qnorm.mean(dim=1, keepdim=True)
        qnorm_std = raw_qnorm.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        normalized_qnorm = (raw_qnorm - qnorm_mean) / qnorm_std
        patch_weights = torch.softmax(normalized_qnorm / self.temperature, dim=1)

        group_size = num_patches // self.context_tokens
        grouped_patches = patch_tokens.reshape(batch_size, self.context_tokens, group_size, dim)
        grouped_weights = patch_weights.reshape(batch_size, self.context_tokens, group_size)
        within_group_weights = grouped_weights / grouped_weights.sum(dim=2, keepdim=True).clamp_min(1e-6)

        context_tokens = (grouped_patches * within_group_weights.unsqueeze(-1)).sum(dim=2)
        query = self.query_norm(cls_token)
        context_tokens = self.context_norm(context_tokens)

        attention_output, attention_weights = self.cross_attention(
            query=query,
            key=context_tokens,
            value=context_tokens,
            need_weights=capture_maps,
            average_attn_weights=False,
        )

        attention_output = self.output_proj(attention_output[:, 0, :])
        alpha = self.get_fusion_alpha().to(
            device=attention_output.device,
            dtype=attention_output.dtype,
        )
        enhanced_cls = cls_token[:, 0, :] + alpha * attention_output

        if capture_maps:
            context_attention = attention_weights.float().mean(dim=1).squeeze(1)
            expanded_attention = context_attention.repeat_interleave(group_size, dim=1)
            combined_contribution = (
                within_group_weights.float() * context_attention.unsqueeze(-1)
            ).reshape(batch_size, num_patches)

            self.last_visualization = {
                "teacher_qnorm": raw_qnorm.cpu(),
                "student_patch_norm": patch_tokens.float().norm(dim=-1).detach().cpu(),
                "pooling_weight": within_group_weights.reshape(batch_size, num_patches).detach().cpu(),
                "context_attention": expanded_attention.detach().cpu(),
                "combined_contribution": combined_contribution.detach().cpu(),
            }
        else:
            self.last_visualization = None

        return enhanced_cls
        
    def get_visualization_maps(self):
        if self.last_visualization is None:
            raise RuntimeError(
                "No visualization maps were captured. "
                "Call forward with capture_maps=True first."
            )
        return {name: value.clone() for name, value in self.last_visualization.items()}

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE


        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate= cfg.MODEL.DROP_OUT,attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, gem_pool=cfg.MODEL.GEM_POOLING, stem_conv=cfg.MODEL.STEM_CONV)
        self.in_planes = self.base.in_planes
        self.qnorm_prior = bool(
            cfg.MODEL.QNORM_PRIOR
        )

        self.qnorm_guided_pooling = None

        if cfg.MODEL.QNORM_PRIOR:
            if cfg.MODEL.JPM:
                raise ValueError(
                    "Q-Norm cross attention requires JPM=False."
                )

            self.qnorm_guided_pooling = (

                QNormContextCrossAttention(
                    dim=self.in_planes,
                    context_tokens=(
                        cfg.MODEL.QNORM_CONTEXT_TOKENS
                    ),
                    num_heads=(
                        cfg.MODEL.QNORM_CROSS_ATTN_HEADS
                    ),
                    temperature=(
                        cfg.MODEL.QNORM_TEMPERATURE
                    ),
                    fusion_alpha=(
                        cfg.MODEL.QNORM_FUSION_ALPHA
                    ),
                    learnable_alpha=(
                        cfg.MODEL.QNORM_LEARNABLE_ALPHA
                    ),
                 )

            )
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path,hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            if self.reduce_feat_dim:
                self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
                self.fcneck.apply(weights_init_xavier)
                self.in_planes = cfg.MODEL.FEAT_DIM
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.dropout = nn.Dropout(self.dropout_rate)

        if pretrain_choice == 'self':
            self.load_param(model_path)

    def forward(self, x, label=None, cam_label= None, view_label=None,teacher_qnorm=None,return_activation_maps=False,):
        #global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        use_qnorm = (
            self.qnorm_prior
            and teacher_qnorm is not None
        )


        if (
        self.qnorm_guided_pooling is not None
        and teacher_qnorm is not None
        ):
            student_tokens, _ = (
                self.base.forward_features(
                    x,
                    camera_id=cam_label,
                    view_id=view_label,
                    return_tokens=True,
                    return_qnorm=False,
                )
            )
            global_feat = self.qnorm_guided_pooling(
                student_tokens,
                teacher_qnorm,
                capture_maps=return_activation_maps,
            )
        else:
            global_feat = self.base(
                x,
                cam_label=cam_label,
                view_label=view_label,
            )
            
        if self.reduce_feat_dim:
            global_feat = self.fcneck(global_feat)
        feat = self.bottleneck(global_feat)
        feat_cls = self.dropout(feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat_cls, label)
            else:
                cls_score = self.classifier(feat_cls)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location = 'cpu')
        for i in param_dict:
            try:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            except:
                continue
        print('Loading pretrained model from {}'.format(trained_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'swin_base_patch4_window7_224': swin_base_patch4_window7_224,
    'swin_small_patch4_window7_224': swin_small_patch4_window7_224,
}

def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
