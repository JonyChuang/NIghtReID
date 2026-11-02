"""Preserve reliable single-layer identity margins in the fused descriptor.

No neighborhood cross-entropy, camera filtering, or layer-alignment loss.
The current fused queries receive gradients; layer references and all candidate
keys are detached. Memory updates are explicit and separate from forward.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class MultiLayerInformationPreservation(nn.Module):
    def __init__(self, labels, dim=768, num_layers=4, min_margin=.05,
                 max_margin=.2, momentum=.2, hard_negatives=5):
        super().__init__()
        if (not all(math.isfinite(float(v)) for v in (min_margin, max_margin, momentum))
                or not 0 <= min_margin < max_margin <= 2 or not 0 <= momentum < 1
                or dim < 1 or num_layers < 1 or hard_negatives < 1):
            raise ValueError('Invalid information preservation settings')
        labels = torch.as_tensor(labels, dtype=torch.long)
        if labels.ndim != 1 or not labels.numel():
            raise ValueError('Preservation requires one identity label per training image')
        self.register_buffer('labels', labels)
        self.register_buffer('features', torch.zeros(len(labels), dim))
        self.register_buffer('layer_features', torch.zeros(len(labels), num_layers, dim))
        self.register_buffer('valid', torch.zeros(len(labels), dtype=torch.bool))
        self.min_margin, self.max_margin = float(min_margin), float(max_margin)
        self.momentum, self.hard_negatives = float(momentum), int(hard_negatives)
        self.num_layers = int(num_layers)

    def validate_batch(self, features, layers, indices, labels):
        if (features.ndim != 2 or features.shape != (len(indices), self.features.shape[1])
                or layers.shape != (len(indices), self.num_layers, self.features.shape[1])
                or indices.ndim != 1 or labels.shape != indices.shape or not indices.numel()):
            raise ValueError('Expected fused [B,D], layers [B,L,D], indices and labels [B]')
        if indices.min() < 0 or indices.max() >= len(self.labels):
            raise ValueError('Training-image index outside preservation memory')
        if not torch.equal(labels, self.labels[indices]):
            raise ValueError('Training-image indices do not match identity labels')

    @torch.no_grad()
    def coalesce(self, features, indices):
        unique, inverse = indices.unique(sorted=True, return_inverse=True)
        normalized = F.normalize(features.detach().float(), dim=-1)
        summed = normalized.new_zeros((len(unique),) + tuple(normalized.shape[1:]))
        summed.index_add_(0, inverse, normalized)
        return unique, F.normalize(summed, dim=-1)

    @torch.no_grad()
    def select_negatives(self, fused_similarity, layer_similarity, negative_mask):
        # Union of the fused and per-layer top-K confusing identities. These
        # choices are detached and all selected candidates must be different-ID.
        scores = torch.cat((fused_similarity.detach()[None], layer_similarity), dim=0)
        count = min(self.hard_negatives, int(negative_mask.sum()))
        selected = scores.masked_fill(~negative_mask[None], -torch.inf).topk(count, dim=1).indices
        return selected.flatten().unique(sorted=True)

    def forward(self, features, layer_features, indices, labels):
        self.validate_batch(features, layer_features, indices, labels)
        with torch.autocast(device_type=features.device.type, enabled=False):
            queries = F.normalize(features.float(), dim=-1)
            unique, current = self.coalesce(queries, indices)
            _, current_layers = self.coalesce(layer_features, indices)
            old = self.valid.clone()
            old[unique] = False
            old_indices = old.nonzero(as_tuple=True)[0]
            candidates = torch.cat((old_indices, unique))
            keys = torch.cat((self.features[old_indices], current))
            layer_keys = torch.cat((self.layer_features[old_indices], current_layers))
            fused_similarity = queries @ keys.T
            with torch.no_grad():
                layer_queries = F.normalize(layer_features.detach().float(), dim=-1)
                layer_similarity = torch.einsum('bld,nld->bln', layer_queries, layer_keys)
                same = labels[:, None] == self.labels[candidates][None, :]
                eligible = indices[:, None] != candidates[None, :]
                positives = same & eligible
                negatives = ~same & eligible
            losses = []
            stats = dict(anchors=len(indices), active_anchors=0, available_positives=int(positives.sum()),
                         candidate_triplets=0, reliable_triplets=0, violated_triplets=0,
                         selected_negatives=0, target_margin_sum=0., fused_margin_sum=0.,
                         candidate_images=len(candidates), memory_images=int(self.valid.sum()))
            for layer in range(self.num_layers):
                stats[f'reference_layer_{layer}'] = 0
            for anchor in range(len(indices)):
                pos = positives[anchor].nonzero(as_tuple=True)[0]
                if not len(pos) or not negatives[anchor].any():
                    continue
                neg = self.select_negatives(fused_similarity[anchor], layer_similarity[anchor], negatives[anchor])
                with torch.no_grad():
                    single = layer_similarity[anchor]
                    layer_margin = single[:, pos, None] - single[:, None, neg]
                    best_margin, best_layer = layer_margin.max(dim=0)
                    reliable = best_margin > self.min_margin
                    target = best_margin.clamp(max=self.max_margin)
                stats['selected_negatives'] += len(neg)
                stats['candidate_triplets'] += len(pos) * len(neg)
                count = int(reliable.sum())
                if not count:
                    continue
                fused_margin = fused_similarity[anchor, pos, None] - fused_similarity[anchor, None, neg]
                gap = target - fused_margin
                # Average all reliable pairs, including already satisfied ones;
                # then average active queries so large identities do not dominate.
                losses.append(F.relu(gap[reliable]).mean())
                stats['active_anchors'] += 1
                stats['reliable_triplets'] += count
                stats['violated_triplets'] += int((gap.detach()[reliable] > 0).sum())
                stats['target_margin_sum'] += float(target[reliable].sum())
                stats['fused_margin_sum'] += float(fused_margin.detach()[reliable].sum())
                for layer in range(self.num_layers):
                    stats[f'reference_layer_{layer}'] += int(((best_layer == layer) & reliable).sum())
            loss = torch.stack(losses).mean() if losses else fused_similarity.sum() * 0.
            return loss, stats

    @torch.no_grad()
    def update(self, features, layer_features, indices, labels):
        """After a successful optimizer step, store this forward pass's features."""
        self.validate_batch(features, layer_features, indices, labels)
        unique, current = self.coalesce(features, indices)
        _, current_layers = self.coalesce(layer_features, indices)
        fused = F.normalize(self.momentum * self.features[unique] + (1 - self.momentum) * current, dim=-1)
        layers = F.normalize(self.momentum * self.layer_features[unique]
                             + (1 - self.momentum) * current_layers, dim=-1)
        self.features[unique] = torch.where(self.valid[unique, None], fused, current)
        self.layer_features[unique] = torch.where(self.valid[unique, None, None], layers, current_layers)
        self.valid[unique] = True
