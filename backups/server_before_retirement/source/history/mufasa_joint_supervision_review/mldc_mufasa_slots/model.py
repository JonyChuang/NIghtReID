"""Jointly trained MLDC + MUFASA, with a single 768D retrieval output.

Method reference: https://github.com/visinf/mufasa/tree/cb4236e7fb37ebd157b7d6d73447565dd2e6f8f0
Independent SA, hard-mask assignment, adjacent-pair means and M-Fusion follow
the reference. MLDC-guided set readout and normalized residual are ReID changes.
"""
import itertools
import math

import torch
from torch import nn
import torch.nn.functional as F
from mldc_cls_readout.model.make_model_mldc import MLDCReID


def residual_feature(global_feature, residual, scale):
    g = F.normalize(global_feature.float(), dim=-1)
    # Do not normalize twice at scale zero: this is the original normalized BN.
    if not torch.is_tensor(scale) and scale == 0:
        return g
    if torch.is_tensor(scale) and scale.ndim == 1:
        scale = scale[:, None]
    return F.normalize(g + scale * F.normalize(residual.float(), dim=-1), dim=-1)


class SlotAttention(nn.Module):
    def __init__(self, count, dim, iterations, hidden):
        super().__init__()
        self.iterations, self.dim = iterations, dim
        self.initial = nn.Parameter(torch.empty(1, count, dim))
        nn.init.xavier_uniform_(self.initial)
        self.input_norm = nn.LayerNorm(768)
        self.slot_norm = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.key = nn.Linear(768, dim, bias=False)
        self.value = nn.Linear(768, dim, bias=False)
        self.query = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def forward(self, patches):
        inputs = self.input_norm(patches)
        keys, values = self.key(inputs) / math.sqrt(self.dim), self.value(inputs)
        initial = self.initial.expand(len(inputs), -1, -1)
        slots = initial
        for i in range(self.iterations):
            # Reference default: bi-level truncation on the final SA iteration.
            if i == self.iterations - 1:
                slots = slots.detach() + initial - initial.detach()
            previous = slots
            query = self.query(self.slot_norm(slots))
            assignment = (keys @ query.transpose(1, 2)).softmax(dim=-1)
            weights = assignment + 1e-8
            weights = weights / weights.sum(dim=1, keepdim=True)
            updates = weights.transpose(1, 2) @ values
            slots = self.gru(updates.reshape(-1, self.dim), previous.reshape(-1, self.dim))
            slots = slots.reshape(len(inputs), -1, self.dim)
            slots = slots + self.mlp(self.mlp_norm(slots))
        return slots, assignment


class MaskAssignment(nn.Module):
    """Exact maximum-IoU linear assignment; K=4 needs only 24 permutations.

This solves the same objective as Hungarian matching, without a SciPy runtime
dependency or a square-grid assumption. Ties use lexicographic slot order.
"""
    def __init__(self, count):
        super().__init__()
        permutations = torch.tensor(list(itertools.permutations(range(count))), dtype=torch.long)
        self.register_buffer('permutations', permutations, persistent=False)
        self.count = count

    @torch.no_grad()
    def forward(self, previous, current):
        a = F.one_hot(previous.argmax(-1), self.count).float().transpose(1, 2)
        b = F.one_hot(current.argmax(-1), self.count).float().transpose(1, 2)
        intersection = a @ b.transpose(1, 2)
        union = a.sum(-1, keepdim=True) + b.sum(-1).unsqueeze(1) - intersection
        iou = torch.where(union > 0, intersection / union.clamp_min(1), 0.)
        rows = torch.arange(self.count, device=iou.device)
        scores = iou[:, rows[None, :], self.permutations].sum(-1)
        order = self.permutations[scores.argmax(-1)]
        before = iou.diagonal(dim1=1, dim2=2).mean(-1)
        after = iou.gather(2, order.unsqueeze(-1)).squeeze(-1).mean(-1)
        return order, before, after


class MultiLayerSlots(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg.MUFASA
        self.match, self.fusion = c.MATCH, c.FUSION
        self.encoders = nn.ModuleList([
            SlotAttention(c.NUM_SLOTS, c.SLOT_DIM, c.ITERATIONS, c.SA_HIDDEN) for _ in c.LAYERS])
        self.assignment = MaskAssignment(c.NUM_SLOTS)
        self.projection = nn.Sequential(nn.Linear(3*c.SLOT_DIM, 3*c.SLOT_DIM), nn.GELU(),
                                        nn.Linear(3*c.SLOT_DIM, c.SLOT_DIM))

    def forward(self, patches):
        pairs = [encoder(p) for encoder, p in zip(self.encoders, patches)]
        slots, maps = [p[0] for p in pairs], [p[1] for p in pairs]
        raw_maps = torch.stack(maps, dim=1)
        orders, before, after = [], [], []
        for i in range(1, len(slots)):
            order, pre, post = self.assignment(maps[i-1], maps[i])
            if not self.match:
                order = torch.arange(slots[i].shape[1], device=slots[i].device).expand(len(slots[i]), -1)
                post = pre
            slots[i] = slots[i].gather(1, order.unsqueeze(-1).expand_as(slots[i]))
            maps[i] = maps[i].gather(2, order.unsqueeze(1).expand_as(maps[i]))
            orders.append(order)
            before.append(pre)
            after.append(post)
        if self.fusion == 'mfusion':
            inputs = torch.cat([(a+b)/2 for a, b in zip(slots[:-1], slots[1:])], dim=-1)
        else:
            # Keep the identical projection and parameter count for the mean control.
            average = torch.stack(slots).mean(0)
            inputs = torch.cat([average, average, average], dim=-1)
        fused = self.projection(inputs)
        aligned_maps = torch.stack(maps, dim=1)
        fused_map = (((aligned_maps[:, :-1] + aligned_maps[:, 1:])/2).mean(1)
                     if self.fusion == 'mfusion' else aligned_maps.mean(1))
        return {'slots': fused, 'layer_slots': torch.stack(slots, dim=1),
                'raw_maps': raw_maps, 'maps': aligned_maps, 'fused_map': fused_map,
                'orders': torch.stack(orders, dim=1),
                'match_before': torch.stack(before, dim=1), 'match_after': torch.stack(after, dim=1)}


class SetReadout(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.global_norm, self.slot_norm = nn.LayerNorm(768), nn.LayerNorm(dim)
        self.query = nn.Linear(768, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, 768, bias=False)
        self.scale = math.sqrt(dim)

    def forward(self, g, slots):
        normalized = self.slot_norm(slots)
        query = self.query(self.global_norm(g))
        attention = torch.einsum('bd,bkd->bk', query, self.key(normalized)) / self.scale
        attention = attention.softmax(-1)
        output = torch.einsum('bk,bkd->bd', attention, self.value(slots))
        return output, attention


class BroadcastDecoder(nn.Module):
    """Training-only positional spatial-broadcast MLP, with four hidden layers.

Slot-specific reconstructions are mixed by decoder mask softmax. Decoder masks
are reconstruction variables, distinct from the SA masks used for matching.
"""
    def __init__(self, dim, count, hidden):
        super().__init__()
        self.position = nn.Parameter(torch.empty(1, 1, count, dim))
        nn.init.normal_(self.position, std=0.02)
        layers = [nn.Linear(dim, hidden), nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, 769))
        self.mlp = nn.Sequential(*layers)

    def forward(self, slots):
        outputs = self.mlp(slots.unsqueeze(2) + self.position)
        masks = outputs[..., -1:].softmax(dim=1)
        return (outputs[..., :768] * masks).sum(1)


class ResidualGate(nn.Module):
    """Image-dependent injection strength, initialized to the old fixed scale."""
    def __init__(self, initial, maximum):
        super().__init__()
        self.maximum = maximum
        self.network = nn.Sequential(nn.Linear(1536, 128), nn.GELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, math.log(initial/(maximum-initial)))

    def forward(self, g, residual):
        inputs = torch.cat([F.normalize(g.float(), dim=-1),
                            F.normalize(residual.float(), dim=-1)], dim=-1)
        return self.maximum * self.network(inputs).sigmoid().squeeze(-1)


class JointMLDCMUFASA(nn.Module):
    def __init__(self, cfg, num_classes, camera_num, view_num):
        super().__init__()
        # Construct the original MLDC first, preserving its initialization RNG
        # sequence and SSL loading behavior before adding the new modules.
        self.mldc_model = MLDCReID(cfg, num_classes, camera_num, view_num)
        self.layers = list(cfg.MUFASA.LAYERS)
        self.scale = float(cfg.MUFASA.RESIDUAL_SCALE)
        self._captured = {}
        self._handles = []
        for layer in sorted(set(self.layers + [12])):
            def capture(module, args, output, number=layer):
                self._captured[number] = output[:, 1:]
            self._handles.append(self.mldc_model.base.blocks[layer-1].register_forward_hook(capture))
        self.slots = MultiLayerSlots(cfg)
        self.readout = SetReadout(cfg.MUFASA.SLOT_DIM)
        self.gate = ResidualGate(self.scale, float(cfg.MUFASA.MAX_RESIDUAL_SCALE))
        self.adaptive_gate = cfg.MUFASA.ADAPTIVE_GATE
        self.classifier = nn.Linear(768, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        grid = self.mldc_model.base.patch_embed
        self.grid = (grid.num_y, grid.num_x)
        self.decoder = BroadcastDecoder(cfg.MUFASA.SLOT_DIM, self.grid[0]*self.grid[1],
                                        cfg.MUFASA.DECODER_HIDDEN)

    def extract_features(self, x, cam_label=None, view_label=None):
        self._captured.clear()
        layers = self.mldc_model.base(x, cam_label=cam_label, view_label=view_label)
        raw = self.mldc_model.mldc(layers)
        g = self.mldc_model.bottleneck(raw)
        patches = [self._captured[l] for l in self.layers]
        # Only the reconstruction target is stopped; SA inputs retain gradients.
        target = self._captured[12].detach()
        self._captured.clear()
        return g, raw, patches, target

    def refine(self, g, patches, target=None, reconstruct=False):
        # Match baseline AMP for the ViT, but keep SA/reconstruction and the
        # normalized residual in FP32. Casting does not detach the ViT graph.
        with torch.autocast(device_type=g.device.type, enabled=False):
            parts = self.slots([p.float() for p in patches])
            r, attention = self.readout(g.float(), parts['slots'])
            scale = self.gate(g, r) if self.adaptive_gate else g.new_full((len(g),), self.scale).float()
            f = residual_feature(g, r, scale)
            parts.update(global_feature=g.float(), residual=r, feature=f,
                         readout_attention=attention, fusion_scale=scale)
            if reconstruct:
                parts['logits'] = self.classifier(f)
                parts['reconstruction'] = self.decoder(parts['slots'])
                parts['target'] = target.detach().float()
        return parts

    def forward(self, x, cam_label=None, view_label=None, return_parts=False, reconstruct=None):
        g, raw, patches, target = self.extract_features(x, cam_label, view_label)
        if reconstruct is None:
            reconstruct = self.training
        parts = self.refine(g, patches, target, reconstruct)
        parts['global_raw'] = raw
        if reconstruct:
            parts['global_logits'] = self.mldc_model.classifier(self.mldc_model.dropout(g))
        return parts if return_parts or self.training else parts['feature']

    def readout_weights(self):
        return self.mldc_model.readout_weights()


def residual_metric_loss(cfg, parts, targets):
    """Cosine-margin loss on r; negatives come from detached MLDC hard candidates."""
    same = targets[:, None].eq(targets[None, :])
    positive_mask = same & ~torch.eye(len(targets), dtype=torch.bool, device=targets.device)
    negative_mask = ~same
    if not positive_mask.any(-1).all() or not negative_mask.any(-1).all():
        raise ValueError('Residual metric requires a PK batch with >=2 identities and >=2 images each')
    with torch.no_grad():
        g = F.normalize(parts['global_feature'].float(), dim=-1)
        candidates = (g @ g.T).masked_fill(~negative_mask, -torch.inf).topk(
            min(cfg.MUFASA.HARD_NEGATIVE_CANDIDATES, len(targets)), dim=-1).indices
        candidate_mask = torch.zeros_like(negative_mask).scatter(1, candidates, True) & negative_mask
    r = F.normalize(parts['residual'].float(), dim=-1)
    similarities = r @ r.T
    positive = similarities.masked_fill(~positive_mask, torch.inf).min(-1).values
    negative = similarities.masked_fill(~candidate_mask, -torch.inf).max(-1).values
    return F.relu(cfg.MUFASA.RESIDUAL_MARGIN + negative-positive).mean()


def compute_loss(cfg, parts, targets):
    from loss.triplet_loss import euclidean_dist, hard_example_mining
    with torch.autocast(device_type=parts['feature'].device.type, enabled=False):
        def reid(logits, feature):
            feature = feature.float()
            positive, negative = hard_example_mining(euclidean_dist(feature, feature), targets)
            return F.cross_entropy(logits.float(), targets), F.softplus(positive-negative).mean()
        global_identity, global_triplet = reid(parts['global_logits'], parts['global_raw'])
        identity, triplet = reid(parts['logits'], parts['feature'])
        reconstruction = F.mse_loss(parts['reconstruction'].float(), parts['target'].float())
        residual_metric = residual_metric_loss(cfg, parts, targets)
        # Preserve the full original global objective; do not average it away.
        total = (cfg.MODEL.ID_LOSS_WEIGHT*global_identity
                 + cfg.MODEL.TRIPLET_LOSS_WEIGHT*global_triplet
                 + cfg.MUFASA.FINAL_LOSS_WEIGHT*(cfg.MODEL.ID_LOSS_WEIGHT*identity
                                                + cfg.MODEL.TRIPLET_LOSS_WEIGHT*triplet)
                 + cfg.MUFASA.RECON_WEIGHT*reconstruction
                 + cfg.MUFASA.RESIDUAL_METRIC_WEIGHT*residual_metric)
    if not torch.isfinite(total):
        raise FloatingPointError('Non-finite MUFASA loss')
    return total, {'global_identity': global_identity, 'global_triplet': global_triplet,
                   'identity': identity, 'triplet': triplet, 'reconstruction': reconstruction,
                   'residual_metric': residual_metric}
