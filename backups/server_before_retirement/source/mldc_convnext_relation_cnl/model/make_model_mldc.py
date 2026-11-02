"""Original MLDC+CNL student with an auxiliary frozen spatial CNN teacher."""
import torch
from yacs.config import CfgNode
from .base_model import MLDCReID as Baseline, add_mldc_config as add_baseline_config
from .base_model import configure_head_learning_rates
from .convnext_teacher import FrozenConvNeXtTeacher
from .patch_relation import patch_relation_loss


def add_mldc_config(cfg):
    cfg = add_baseline_config(cfg)
    cfg.PATCH_KD = CfgNode(dict(ENABLED=True,
        TEACHER_PATH='pretrain_model/convnext_tiny_1k_224_ema.pth', TEACHER_STAGE=3,
        STUDENT_LAYER=12, WEIGHT=.1, RAMP_EPOCHS=20))
    return cfg


class MLDCReID(Baseline):
    def __init__(self, cfg, num_classes, camera_num, view_num, initialize=True,
                 teacher_override=None):
        super().__init__(cfg, num_classes, camera_num, view_num,
                         initialize=initialize and cfg.MODEL.PRETRAIN_CHOICE == 'imagenet')
        self.kd_settings = cfg.PATCH_KD.clone()
        self.kd_enabled = bool(cfg.PATCH_KD.ENABLED)
        self.texture_teacher = None  # shared processor checkpoint metadata API
        if self.kd_settings.TEACHER_STAGE != 3 or self.kd_settings.STUDENT_LAYER != 12:
            raise ValueError('This independent experiment aligns CNN stage 3 with ViT block 12')
        if (list(cfg.INPUT.SIZE_TRAIN) != [256, 128] or list(cfg.MODEL.STRIDE_SIZE) != [16, 16]
                or cfg.MODEL.MLDC_TRAIN_FEATURE != 'mldc' or cfg.TEST.MLDC_RETRIEVAL != 'mldc'
                or cfg.TEST.NECK_FEAT != 'after'):
            raise ValueError('Keep the original 256x128, stride16, BN-after MLDC retrieval path')
        if cfg.PATCH_KD.WEIGHT < 0 or cfg.PATCH_KD.RAMP_EPOCHS < 1:
            raise ValueError('Invalid patch KD weight/ramp')
        if self.kd_enabled and cfg.MODEL.PRETRAIN_CHOICE != 'self':
            # Teacher initialization must not change student initialization or its RNG stream.
            with torch.random.fork_rng(devices=[]):
                self.texture_teacher = teacher_override or FrozenConvNeXtTeacher(
                    cfg.PATCH_KD.TEACHER_PATH, cfg.PATCH_KD.TEACHER_STAGE,
                    cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD, load_weights=initialize)
            self.texture_teacher.requires_grad_(False).eval()
        if initialize and cfg.MODEL.PRETRAIN_CHOICE == 'self':
            self.load_param(cfg.MODEL.PRETRAIN_PATH)

    def train(self, mode=True):
        super().train(mode)
        if self.texture_teacher is not None:
            self.texture_teacher.eval()
        return self

    def forward(self, x, label=None, cam_label=None, view_label=None, return_components=False,
                return_cnl_features=False, return_distillation=False, sample_indices=None):
        if not self.training or not return_distillation:
            return super().forward(x, label, cam_label, view_label, return_components, return_cnl_features)
        captured = []
        handle = None
        if self.kd_enabled:
            if self.texture_teacher is None:
                raise RuntimeError('Teacher-free inference checkpoint cannot be used to resume training')
            handle = self.base.blocks[self.kd_settings.STUDENT_LAYER - 1].register_forward_hook(
                lambda module, args, output: captured.append(output))
        try:
            result = super().forward(x, label, cam_label, view_label, return_components, return_cnl_features)
        finally:
            if handle is not None:
                handle.remove()
        if not self.kd_enabled:
            return (*result, dict(relation=result[1].sum() * 0., stats={}))
        tokens = captured.pop()[:, 1:]  # exclude CLS; no new projection heads
        if tokens.shape[1] != 128:
            raise ValueError('Expected the unchanged 16x8 student patch grid')
        with torch.no_grad():
            teacher_map = self.texture_teacher(x.detach())
        if teacher_map.shape[-2:] != (16, 8):
            raise ValueError('Expected teacher stage-3 grid 16x8')
        teacher_tokens = teacher_map.flatten(2).transpose(1, 2)
        loss, stats = patch_relation_loss(tokens, teacher_tokens)
        return (*result, dict(relation=loss, stats=stats))

    def distillation_loss(self, terms, epoch):
        settings = self.kd_settings
        ramp = min(1., max(0., float(epoch) / settings.RAMP_EPOCHS))
        return float(settings.WEIGHT) * ramp * terms['relation']

    def architecture(self):
        return dict(variant='convnext_patch_relation_20261027', descriptor_dim=768,
            enabled=self.kd_enabled, teacher='official ImageNet-1K ConvNeXt-T, frozen',
            teacher_stage=self.kd_settings.TEACHER_STAGE, student_layer=self.kd_settings.STUDENT_LAYER,
            patch_grid=[16, 8], objective='mean off-diagonal cosine-Gram MSE',
            weight=float(self.kd_settings.WEIGHT), ramp_epochs=self.kd_settings.RAMP_EPOCHS,
            inference='original ViT + MLDC + BNneck only')

    def inference_state_dict(self):
        return {key: value for key, value in self.state_dict().items()
                if not key.startswith('texture_teacher.')}

    def load_param(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint.get('architecture') != self.architecture():
            raise ValueError('Checkpoint architecture differs from this independent experiment')
        self.load_state_dict(checkpoint['state_dict'], strict=True)
        self.checkpoint_epoch = int(checkpoint.get('epoch', 0))


def make_model(cfg, num_class, camera_num, view_num, initialize=True):
    return MLDCReID(cfg, num_class, camera_num, view_num, initialize=initialize)
