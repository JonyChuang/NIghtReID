"""PID-free same-layer neighbor mining with a shared four-layer memory.

Each layer uses its own relative threshold; the image-index union supervises
all four layers. Forward never mutates the memory. Update only after a
successful optimizer step, using the same features as that forward pass.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SharedLayerCNL(nn.Module):
    def __init__(self, cameras, dim=768, num_layers=4, gamma=.9,
                 temperature=.05, momentum=.2, cross_camera=True):
        super().__init__()
        if (not all(math.isfinite(float(x)) for x in (gamma, temperature, momentum))
                or not 0 < gamma <= 1 or temperature <= 0 or not 0 <= momentum < 1
                or dim < 1 or num_layers != 4):
            raise ValueError('Use four layers, positive temperature, gamma in (0,1], momentum in [0,1)')
        cameras = torch.as_tensor(cameras, dtype=torch.long)
        if cameras.ndim != 1 or not cameras.numel():
            raise ValueError('Expected one camera identifier per training image')
        self.register_buffer('cameras', cameras.clone())
        self.register_buffer('features', torch.zeros(len(cameras), num_layers, dim))
        self.register_buffer('valid', torch.zeros(len(cameras), dtype=torch.bool))
        self.gamma, self.temperature, self.momentum = float(gamma), float(temperature), float(momentum)
        self.cross_camera, self.num_layers = bool(cross_camera), int(num_layers)

    def validate_batch(self, layers, indices, cameras):
        if (layers.ndim != 3 or layers.shape[1:] != self.features.shape[1:]
                or indices.ndim != 1 or not indices.numel() or indices.dtype != torch.long
                or cameras.shape != indices.shape or layers.shape[0] != len(indices)):
            raise ValueError('Expected layers [B,4,D], image indices [B] and cameras [B]')
        if indices.min() < 0 or indices.max() >= len(self.cameras):
            raise ValueError('Image index outside shared memory')
        if not torch.equal(cameras, self.cameras[indices]):
            raise ValueError('Image index/camera metadata mismatch')

    @torch.no_grad()
    def coalesce(self, layers, indices):
        unique, inverse = indices.unique(sorted=True, return_inverse=True)
        normalized = F.normalize(layers.detach().float(), dim=-1)
        summed = normalized.new_zeros((len(unique),) + tuple(normalized.shape[1:]))
        summed.index_add_(0, inverse, normalized)
        return unique, F.normalize(summed, dim=-1)

    def forward(self, layers, indices, cameras):
        self.validate_batch(layers, indices, cameras)
        with torch.autocast(device_type=layers.device.type, enabled=False):
            queries = F.normalize(layers.float(), dim=-1)
            unique, current = self.coalesce(layers, indices)
            old = self.valid.clone()
            old[unique] = False
            old_indices = old.nonzero(as_tuple=True)[0]
            candidates = torch.cat((old_indices, unique))
            # Current images replace their stale slots; repeated augmentations
            # are one candidate image with four separate, normalized keys.
            keys = torch.cat((self.features[old_indices], current))
            scores = torch.einsum('bld,nld->bln', queries, keys)
            with torch.no_grad():
                eligible = indices[:, None] != candidates[None, :]
                if self.cross_camera:
                    eligible &= cameras[:, None] != self.cameras[candidates][None, :]
                maximum = scores.detach().masked_fill(~eligible[:, None], -torch.inf).amax(-1, keepdim=True)
                discovered = (scores.detach() >= self.gamma * maximum) & eligible[:, None] & maximum.isfinite()
                support = discovered.sum(1)
                positives = support > 0
                negatives = eligible & ~positives  # Candidate negatives; no identity filtering.
                active = positives.any(1) & negatives.any(1)
            if active.any():
                logits = scores[active] / self.temperature
                denominator = torch.logsumexp(logits.masked_fill(~eligible[active, None], -torch.inf), dim=-1, keepdim=True)
                log_probability = logits - denominator
                positive_log = log_probability.masked_fill(~positives[active, None], 0.)
                per_layer = -positive_log.sum(-1) / positives[active].sum(-1, keepdim=True)
                loss = per_layer.mean()  # Equal layers, then equal active query images.
            else:
                loss = scores.sum() * 0.
            stats = dict(anchors=len(indices), active_anchors=int(active.sum()),
                empty_positive_anchors=int((~positives.any(1)).sum()),
                no_negative_anchors=int((positives.any(1) & ~negatives.any(1)).sum()),
                selected_positives=int(positives.sum()), eligible_pairs=int(eligible.sum()),
                candidate_images=len(candidates), memory_images=int(self.valid.sum()))
            for layer in range(self.num_layers):
                stats[f'discovered_layer_{layer + 9}'] = int(discovered[:, layer].sum())
            for count in range(1, self.num_layers + 1):
                stats[f'support_{count}'] = int((support == count).sum())
            return loss, stats

    @torch.no_grad()
    def update(self, layers, indices, cameras):
        self.validate_batch(layers, indices, cameras)
        unique, current = self.coalesce(layers, indices)
        mixed = F.normalize(self.momentum * self.features[unique]
                            + (1 - self.momentum) * current, dim=-1)
        self.features[unique] = torch.where(self.valid[unique, None, None], mixed, current)
        self.valid[unique] = True


def create_shared_memory(cfg, model, dataset, device):
    if cfg.SOLVER.CNL_WEIGHT == 0:
        return None
    # Only the camera column is read. Ground-truth PID stays outside CNL.
    return SharedLayerCNL([int(row[2]) for row in dataset.entries],
        dim=model.base.in_planes, num_layers=len(model.mldc.selected_layers),
        gamma=cfg.SOLVER.CNL_GAMMA, temperature=cfg.SOLVER.CNL_TEMPERATURE,
        momentum=cfg.SOLVER.CNL_MOMENTUM, cross_camera=cfg.SOLVER.CNL_CROSS_CAMERA).to(device)
