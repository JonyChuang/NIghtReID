"""MUFASA-inspired frozen increment, with a single 768D retrieval output.

Method reference: https://github.com/visinf/mufasa/tree/cb4236e7fb37ebd157b7d6d73447565dd2e6f8f0
Independent SA, hard-mask assignment, adjacent-pair means and M-Fusion follow
the reference. MLDC-guided set readout and normalized residual are ReID changes.
"""
import itertools
import math

import torch
from torch import nn
import torch.nn.functional as F
from yacs.config import CfgNode as CN
from mldc_cls_readout.model.make_model_mldc import MLDCReID


def residual_feature(global_feature, residual, scale):
    g = F.normalize(global_feature.float(), dim=-1)
    # Do not normalize twice at scale zero: this is the original normalized BN.
    if scale == 0:
        return g
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


class FrozenMLDCMUFASA(nn.Module):
    def __init__(self, cfg, num_classes, camera_num, view_num):
        super().__init__()
        checkpoint = torch.load(cfg.MUFASA.BASELINE, map_location='cpu', weights_only=False)
        if checkpoint.get('epoch') != 120 or checkpoint.get('readout') != 'mldc':
            raise ValueError('Initialize from the independent MLDC epoch-120 checkpoint')
        # Complete saved configs are loaded directly: YACS merge would decode
        # the string DEVICE_ID='0' a second time and turn it into integer 0.
        original = CN.load_cfg(checkpoint['config'])
        for name in ('SIZE_TRAIN', 'SIZE_TEST', 'PIXEL_MEAN', 'PIXEL_STD', 'PROB', 'RE_PROB', 'PADDING'):
            if getattr(cfg.INPUT, name) != getattr(original.INPUT, name):
                raise ValueError('Original input preprocessing must be retained: ' + name)
        if checkpoint['state_dict']['classifier.weight'].shape[0] != num_classes:
            raise ValueError('Training identity count differs from the MLDC checkpoint')
        original.MODEL.PRETRAIN_CHOICE = 'self'
        original.MODEL.PRETRAIN_PATH = cfg.MUFASA.BASELINE
        original.TEST.NECK_FEAT = 'after'
        self.baseline = MLDCReID(original, num_classes, camera_num, view_num)
        self.baseline.requires_grad_(False)
        self.baseline.eval()
        self.baseline_metrics = checkpoint['metrics']
        del checkpoint
        self.layers = list(cfg.MUFASA.LAYERS)
        self.scale = float(cfg.MUFASA.RESIDUAL_SCALE)
        self._captured = {}
        self._handles = []
        for layer in sorted(set(self.layers + [12])):
            def capture(module, args, output, number=layer):
                self._captured[number] = output[:, 1:].detach()
            self._handles.append(self.baseline.base.blocks[layer-1].register_forward_hook(capture))
        self.slots = MultiLayerSlots(cfg)
        self.readout = SetReadout(cfg.MUFASA.SLOT_DIM)
        self.classifier = nn.Linear(768, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        grid = self.baseline.base.patch_embed
        self.grid = (grid.num_y, grid.num_x)
        self.decoder = BroadcastDecoder(cfg.MUFASA.SLOT_DIM, self.grid[0]*self.grid[1],
                                        cfg.MUFASA.DECODER_HIDDEN)

    def train(self, mode=True):
        super().train(mode)
        self.baseline.eval()
        return self

    def extract_frozen(self, x, cam_label=None, view_label=None):
        self._captured.clear()
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            g = self.baseline(x.float(), cam_label=cam_label, view_label=view_label)
        patches = [self._captured[l] for l in self.layers]
        target = self._captured[12]
        self._captured.clear()
        return g, patches, target

    def refine(self, g, patches, target=None, reconstruct=False):
        parts = self.slots(patches)
        r, attention = self.readout(g, parts['slots'])
        f = residual_feature(g, r, self.scale)
        parts.update(global_feature=g, residual=r, feature=f, readout_attention=attention)
        if reconstruct:
            parts['logits'] = self.classifier(f)
            parts['reconstruction'] = self.decoder(parts['slots'])
            parts['target'] = target.detach()
        return parts

    def forward(self, x, cam_label=None, view_label=None, return_parts=False, reconstruct=None):
        g, patches, target = self.extract_frozen(x, cam_label, view_label)
        if reconstruct is None:
            reconstruct = self.training
        parts = self.refine(g, patches, target, reconstruct)
        return parts if return_parts or self.training else parts['feature']


def compute_loss(cfg, parts, targets):
    from loss.triplet_loss import euclidean_dist, hard_example_mining
    f = parts['feature'].float()
    positive, negative = hard_example_mining(euclidean_dist(f, f), targets)
    identity = F.cross_entropy(parts['logits'].float(), targets)
    triplet = F.softplus(positive-negative).mean()
    reconstruction = F.mse_loss(parts['reconstruction'].float(), parts['target'].float())
    total = (cfg.MODEL.ID_LOSS_WEIGHT*identity + cfg.MODEL.TRIPLET_LOSS_WEIGHT*triplet
             + cfg.MUFASA.RECON_WEIGHT*reconstruction)
    if not torch.isfinite(total):
        raise FloatingPointError('Non-finite MUFASA loss')
    return total, {'identity': identity, 'triplet': triplet, 'reconstruction': reconstruction}
