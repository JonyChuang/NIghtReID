"""Training-image cache of pre-BN MLDC features and epoch-wise reciprocal graphs.

One slot per original training image. The graph is label/camera blind and is
computed in blocks over all valid cached images, never an N-by-N allocation.
Cached vectors are context tokens, not optimization variables.
"""
import torch
from torch import nn
from torch.nn import functional as F


class NeighborMemory(nn.Module):
    def __init__(self, size, dim=768, candidate_k=32, neighbors=8,
                 warmup_epochs=20, block_size=512):
        super().__init__()
        if size < 1 or dim < 1 or not 1 <= neighbors <= candidate_k:
            raise ValueError('Require a positive cache and 1 <= neighbors <= candidate_k')
        if warmup_epochs < 0 or block_size < 1:
            raise ValueError('Invalid graph schedule or block size')
        self.candidate_k = int(candidate_k)
        self.neighbors = int(neighbors)
        self.warmup_epochs = int(warmup_epochs)
        self.block_size = int(block_size)
        self.graph_epoch = 0
        self.register_buffer('features', torch.zeros(size, dim))
        self.register_buffer('valid', torch.zeros(size, dtype=torch.bool))
        self.register_buffer('last_update', torch.full((size,), -1, dtype=torch.long))
        self.register_buffer('neighbor_indices', torch.full((size, neighbors), -1, dtype=torch.long))

    @torch.no_grad()
    def update(self, features, indices, step, applied=True):
        """Commit only successfully optimized original rows; coalesce duplicates."""
        if not applied:
            return
        if features.shape != (len(indices), self.features.shape[1]):
            raise ValueError('Features and stable training-image indices do not align')
        if not len(indices):
            return
        if indices.min() < 0 or indices.max() >= len(self.valid):
            raise IndexError('Training-image index outside neighbor cache')
        if not torch.isfinite(features).all():
            raise FloatingPointError('Non-finite feature in neighbor cache update')
        unique, inverse, counts = indices.unique(sorted=True, return_inverse=True, return_counts=True)
        values = torch.zeros(len(unique), features.shape[1], device=features.device, dtype=torch.float32)
        values.index_add_(0, inverse, features.detach().float())
        values /= counts[:, None]
        self.features[unique] = values
        self.valid[unique] = values.norm(dim=1) > 0
        self.last_update[unique] = int(step)

    @torch.no_grad()
    def refresh_graph(self, epoch):
        """Snapshot reciprocal top-k once per epoch, starting after warmup."""
        self.neighbor_indices.fill_(-1)
        self.graph_epoch = int(epoch)
        ids = self.valid.nonzero(as_tuple=True)[0]
        if epoch <= self.warmup_epochs or len(ids) < 2:
            return
        with torch.autocast(device_type=self.features.device.type, enabled=False):
            keys = F.normalize(self.features[ids].float(), dim=1)
            k = min(self.candidate_k, len(ids) - 1)
            nearest = torch.empty(len(ids), k, dtype=torch.long, device=ids.device)
            for start in range(0, len(ids), self.block_size):
                stop = min(start + self.block_size, len(ids))
                similarity = keys[start:stop] @ keys.T
                rows = torch.arange(stop - start, device=ids.device)
                similarity[rows, torch.arange(start, stop, device=ids.device)] = -torch.inf
                nearest[start:stop] = similarity.topk(k, dim=1, sorted=True).indices
            # nearest contains local indices; retain the most similar reciprocal
            # entries in their original rank order, without filling nonreciprocal slots.
            for start in range(0, len(ids), self.block_size):
                stop = min(start + self.block_size, len(ids))
                candidates = nearest[start:stop]
                owner = torch.arange(start, stop, device=ids.device)[:, None, None]
                mutual = (nearest[candidates] == owner).any(dim=2)
                ranks = torch.arange(k, device=ids.device).expand(stop - start, -1)
                order = ranks.masked_fill(~mutual, k).argsort(dim=1)
                take = min(self.neighbors, k)
                order = order[:, :take]
                chosen = ids[candidates.gather(1, order)]
                chosen = chosen.masked_fill(~mutual.gather(1, order), -1)
                self.neighbor_indices[ids[start:stop], :take] = chosen

    @torch.no_grad()
    def select(self, indices, epoch, step=0):
        """Return active anchor rows, detached context and padding mask."""
        empty = indices.new_empty(0)
        stats = dict(anchors=len(indices), active_anchors=0, support_images=0,
                     cache_images=int(self.valid.sum()), age_sum=0, cache_graph_epoch=self.graph_epoch,
                     single_neighbor_anchors=0, full_neighbor_anchors=0,
                     same_id_neighbors=0, cross_camera_neighbors=0)
        if epoch <= self.warmup_epochs:
            return empty, self.features.new_empty((0, self.neighbors, self.features.shape[1])), self.valid.new_empty((0, self.neighbors)), stats
        if self.graph_epoch != int(epoch):
            raise RuntimeError('Refresh the neighbor graph before each active training epoch')
        selected = self.neighbor_indices[indices]
        valid = (selected >= 0) & (selected != indices[:, None])
        rows = valid.any(1).nonzero(as_tuple=True)[0]
        selected, valid = selected[rows], valid[rows]
        context = self.features[selected.clamp_min(0)].clone().detach()
        context.masked_fill_(~valid[:, :, None], 0.)
        counts = valid.sum(1)
        stats.update(active_anchors=len(rows), support_images=int(counts.sum()),
                     age_sum=int(((int(step) - self.last_update[selected.clamp_min(0)]) * valid).sum()),
                     single_neighbor_anchors=int((counts == 1).sum()),
                     full_neighbor_anchors=int((counts == self.neighbors).sum()))
        return rows, context, ~valid, stats

    @torch.no_grad()
    def diagnostic_relations(self, indices, rows, padding, labels, cameras):
        """Annotations are used only AFTER graph selection, for logging."""
        selected = self.neighbor_indices[indices[rows]].clamp_min(0)
        valid = ~padding
        anchors = indices[rows, None]
        return dict(same_id_neighbors=int(((labels[anchors] == labels[selected]) & valid).sum()),
                    cross_camera_neighbors=int(((cameras[anchors] != cameras[selected]) & valid).sum()))
