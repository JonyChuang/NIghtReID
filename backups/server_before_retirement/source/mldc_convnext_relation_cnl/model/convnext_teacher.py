"""Frozen official ConvNeXt-T. Weight files are supplied locally, never fetched."""
from pathlib import Path
import torch
from torch import nn
from .convnext_official.convnext import ConvNeXt

WEIGHTS_URL = 'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth'


class FrozenConvNeXtTeacher(nn.Module):
    def __init__(self, path, stage=3, input_mean=(.5, .5, .5), input_std=(.5, .5, .5),
                 load_weights=True):
        super().__init__()
        if stage != 3:
            raise ValueError('This experiment uses ConvNeXt stage 3, stride 16')
        self.stage = stage
        self.network = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                                num_classes=1000, drop_path_rate=0.)
        for name, value in [('input_mean', input_mean), ('input_std', input_std),
                            ('imagenet_mean', [.485, .456, .406]),
                            ('imagenet_std', [.229, .224, .225])]:
            self.register_buffer(name, torch.tensor(value).reshape(1, 3, 1, 1), persistent=False)
        self.provenance = dict(architecture='official ConvNeXt-T', source='ImageNet-1K',
            weights_url=WEIGHTS_URL, checkpoint=str(path), loaded=False,
            nightreid_finetuned=False, frozen=True, stage=stage)
        if load_weights:
            checkpoint_path = Path(path)
            if not checkpoint_path.is_file():
                raise FileNotFoundError('Place the official ImageNet-1K EMA checkpoint at {}. '
                                        'Download manually: {}'.format(path, WEIGHTS_URL))
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=True)
            if not isinstance(checkpoint, dict) or 'model' not in checkpoint:
                raise ValueError('Expected the official ConvNeXt checkpoint with a model state_dict')
            self.network.load_state_dict(checkpoint['model'], strict=True)
            self.provenance['loaded'] = True
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        # Remain in evaluation mode even when the student calls model.train().
        return super().train(False)

    def normalize_input(self, images):
        rgb = images.float() * self.input_std + self.input_mean
        return (rgb - self.imagenet_mean) / self.imagenet_std

    @torch.no_grad()
    def forward(self, images):
        # Same already-augmented image as the student; no resize/crop is added.
        x = self.normalize_input(images)
        for index in range(self.stage):
            x = self.network.downsample_layers[index](x)
            x = self.network.stages[index](x)
        return x
