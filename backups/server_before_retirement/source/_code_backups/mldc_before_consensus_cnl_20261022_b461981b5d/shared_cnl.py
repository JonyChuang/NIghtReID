"""PID-free multilayer mining, contrasted in the final retrieval space.

Each layer uses its own relative threshold; their image-index union supervises
the fused BN-after descriptor. A global image index synchronizes its fused key
and four mining keys. Forward never mutates this online memory.
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
        self.register_buffer('fused_features', torch.zeros(len(cameras), dim))
        self.register_buffer('valid', torch.zeros(len(cameras), dtype=torch.bool))
        self.gamma, self.temperature, self.momentum = float(gamma), float(temperature), float(momentum)
        self.cross_camera, self.num_layers = bool(cross_camera), int(num_layers)

    def validate_batch(self, fused, layers, indices, cameras):
        if (layers.ndim != 3 or layers.shape[1:] != self.features.shape[1:]
                or indices.ndim != 1 or not indices.numel() or indices.dtype != torch.long
                or cameras.shape != indices.shape or layers.shape[0] != len(indices)
                or fused.shape != (len(indices), self.fused_features.shape[1])):
            raise ValueError('Expected fused [B,D], layers [B,4,D], image indices [B] and cameras [B]')
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

    def forward(self, fused, layers, indices, cameras):
        self.validate_batch(fused, layers, indices, cameras)
        with torch.autocast(device_type=fused.device.type, enabled=False):
            queries = F.normalize(fused.float(), dim=-1)
            unique, current = self.coalesce(layers, indices)
            _, current_fused = self.coalesce(fused, indices)
            old = self.valid.clone()
            old[unique] = False
            old_indices = old.nonzero(as_tuple=True)[0]
            candidates = torch.cat((old_indices, unique))
            # Current images replace their stale slots; repeated augmentations
            # are one candidate image with four mining keys and one fused key.
            keys = torch.cat((self.features[old_indices], current))
            fused_keys = torch.cat((self.fused_features[old_indices], current_fused))
            with torch.no_grad():
                scores = torch.einsum('bld,nld->bln', F.normalize(layers.detach().float(), dim=-1), keys)
                eligible = indices[:, None] != candidates[None, :]
                if self.cross_camera:
                    eligible &= cameras[:, None] != self.cameras[candidates][None, :]
                maximum = scores.masked_fill(~eligible[:, None], -torch.inf).amax(-1, keepdim=True)
                discovered = (scores >= self.gamma * maximum) & eligible[:, None] & maximum.isfinite()
                support = discovered.sum(1)
                positives = support > 0
                negatives = eligible & ~positives  # Candidate negatives; no identity filtering.
                active = positives.any(1) & negatives.any(1)
            retrieval_scores = queries @ fused_keys.T
            if active.any():
                logits = retrieval_scores[active] / self.temperature
                denominator = torch.logsumexp(logits.masked_fill(~eligible[active], -torch.inf), dim=-1, keepdim=True)
                log_probability = logits - denominator
                positive_log = log_probability.masked_fill(~positives[active], 0.)
                loss = (-positive_log.sum(-1) / positives[active].sum(-1)).mean()
            else:
                loss = retrieval_scores.sum() * 0.
            stats = dict(anchors=len(indices), active_anchors=int(active.sum()),
                empty_positive_anchors=int((~positives.any(1)).sum()),
                no_negative_anchors=int((positives.any(1) & ~negatives.any(1)).sum()),
                selected_positives=int(positives.sum()), eligible_pairs=int(eligible.sum()),
                candidate_images=len(candidates), memory_images=int(self.valid.sum()),
                out_of_batch_candidates=len(old_indices),
                out_of_batch_positives=int(positives[:, :len(old_indices)].sum()))
            for layer in range(self.num_layers):
                stats[f'discovered_layer_{layer + 9}'] = int(discovered[:, layer].sum())
            for count in range(1, self.num_layers + 1):
                stats[f'support_{count}'] = int((support == count).sum())
            return loss, stats

    @torch.no_grad()
    def update(self, fused, layers, indices, cameras):
        self.validate_batch(fused, layers, indices, cameras)
        unique, current = self.coalesce(layers, indices)
        _, current_fused = self.coalesce(fused, indices)
        mixed = F.normalize(self.momentum * self.features[unique]
                            + (1 - self.momentum) * current, dim=-1)
        mixed_fused = F.normalize(self.momentum * self.fused_features[unique]
                                  + (1 - self.momentum) * current_fused, dim=-1)
        self.features[unique] = torch.where(self.valid[unique, None, None], mixed, current)
        self.fused_features[unique] = torch.where(self.valid[unique, None], mixed_fused, current_fused)
        self.valid[unique] = True


def create_shared_memory(cfg, model, dataset, device):
    if cfg.SOLVER.CNL_WEIGHT == 0:
        return None
    # Only the camera column is read. Ground-truth PID stays outside CNL.
    return SharedLayerCNL([int(row[2]) for row in dataset.entries],
        dim=model.base.in_planes, num_layers=len(model.mldc.selected_layers),
        gamma=cfg.SOLVER.CNL_GAMMA, temperature=cfg.SOLVER.CNL_TEMPERATURE,
        momentum=cfg.SOLVER.CNL_MOMENTUM, cross_camera=cfg.SOLVER.CNL_CROSS_CAMERA).to(device)
