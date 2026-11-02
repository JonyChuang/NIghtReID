"""Training-only support-set fusion; all projections belong to the original ViT."""
import torch
import torch.nn.functional as F


def cross_defaults():
    from yacs.config import CfgNode
    return CfgNode(dict(ENABLED=False, MIN_SUPPORTS=2, MAX_SUPPORTS=3,
        REQUIRE_CROSS_CAMERA=True, GROUP_ID_WEIGHT=1., IP_WEIGHT=.1,
        WARMUP_EPOCHS=20, RAMP_EPOCHS=5, FUSION='cross_attention'))


def activation_factor(epoch, settings):
    return min(1., max(0., (int(epoch)-settings.WARMUP_EPOCHS)/settings.RAMP_EPOCHS))


def support_groups(labels, cameras, indices, epoch, settings):
    """One rotating source-image anchor per ID; remove every copy from supports."""
    yy, cc, ii = [value.detach().cpu().tolist() for value in (labels, cameras, indices)]
    identities, seen = {}, {}
    for position, (label, camera, image) in enumerate(zip(yy, cc, ii)):
        if image in seen:
            if (label, camera) != seen[image][:2]:
                raise ValueError('Inconsistent metadata for one source image')
            continue
        seen[image] = (label, camera, position)
        identities.setdefault(label, []).append(position)
    anchors, supports, mapping = [], [], [-1]*len(ii)
    for members in identities.values():
        if len(members) < settings.MIN_SUPPORTS+1:
            continue
        candidates = sorted((a for a in members if not settings.REQUIRE_CROSS_CAMERA
                             or any(cc[b] != cc[a] for b in members)), key=lambda a: ii[a])
        if not candidates:
            continue
        anchor = candidates[(int(epoch)-settings.WARMUP_EPOCHS-1) % len(candidates)]
        others = sorted((b for b in members if b != anchor), key=lambda b: (cc[b] == cc[anchor], ii[b]))
        selected = others[:settings.MAX_SUPPORTS]
        group = len(anchors)
        anchors.append(anchor)
        supports.append(selected)
        for p, image in enumerate(ii):
            if image == ii[anchor]:
                mapping[p] = group
    maximum = max(map(len, supports), default=settings.MIN_SUPPORTS)
    # Padding is masked as attention keys and excluded from CLS/feature means.
    valid = [[True]*len(s)+[False]*(maximum-len(s)) for s in supports]
    padded = [s+[s[0]]*(maximum-len(s)) for s in supports]
    device = labels.device
    return (torch.tensor(anchors, dtype=torch.long, device=device),
            torch.tensor(padded, dtype=torch.long, device=device).reshape(-1, maximum),
            torch.tensor(valid, dtype=torch.bool, device=device).reshape(-1, maximum),
            torch.tensor(mapping, dtype=torch.long, device=device),
            dict(unique_images=len(seen), eligible_ids=len(identities), groups=len(anchors),
                 active_anchors=sum(v >= 0 for v in mapping), support_images=sum(map(len, supports)),
                 crosscam_groups=sum(any(cc[b] != cc[a] for b in s) for a, s in zip(anchors, supports))))


def member_mean(values, supports, valid):
    members = values[supports]
    weights = valid.to(members.dtype)
    return (members*weights[..., None]).sum(1)/weights.sum(1, keepdim=True)


def shared_cross_block(block, cls, keys, key_valid):
    """One CLS query, support patch K/V; reuse LN/QKV/output/FFN/DropPath."""
    query, spatial = block.norm1(cls), block.norm1(keys)
    attention = block.attn
    dim, heads = cls.shape[-1], attention.num_heads
    weight, bias = attention.qkv.weight, attention.qkv.bias
    q = F.linear(query, weight[:dim], None if bias is None else bias[:dim])
    kv = F.linear(spatial, weight[dim:], None if bias is None else bias[dim:])
    q = q.reshape(len(cls), 1, heads, dim//heads).transpose(1, 2)
    kv = kv.reshape(len(cls), keys.shape[1], 2, heads, dim//heads).permute(2, 0, 3, 1, 4)
    logits = (q @ kv[0].transpose(-2, -1))*attention.scale
    logits = logits.masked_fill(~key_valid[:, None, None, :], float('-inf'))
    probabilities = logits.softmax(-1)
    update = (attention.attn_drop(probabilities) @ kv[1]).transpose(1, 2).reshape(len(cls), 1, dim)
    cls = cls+block.drop_path(attention.proj_drop(attention.proj(update)))
    cls = cls+block.drop_path(block.mlp(block.norm2(cls)))
    return cls


def fuse(model, raw_layers, patches, supports, valid):
    settings = model.mvi_cross_settings
    if settings.FUSION == 'mean':
        # Same raw-feature readout and BN as learned fusion, for a matched control.
        return model.mldc(torch.stack([member_mean(layer, supports, valid) for layer in raw_layers]))
    cls = member_mean(raw_layers[7], supports, valid).unsqueeze(1)
    group_cls = []
    for level in range(9, 13):
        spatial = patches[level-1][supports]
        keys = spatial.reshape(len(supports), -1, spatial.shape[-1])
        key_valid = valid[:, :, None].expand(-1, -1, spatial.shape[2]).reshape(len(supports), -1)
        cls = shared_cross_block(model.base.blocks[level-1], cls, keys, key_valid)
        group_cls.append(cls[:, 0])
    unused = cls.new_zeros((8, len(cls), cls.shape[-1]))
    return model.mldc(torch.cat((unused, torch.stack(group_cls)), dim=0))


def cross_terms(model, raw_layers, patches, grouping, mixed, unit, labels, factor):
    zero = unit.sum()*0.
    anchors, supports, valid, mapping, stats = grouping
    result = dict(group_id=zero, ip=zero, weighted=zero, factor=factor, stats=stats)
    if factor == 0. or not len(anchors):
        return result
    group_mixed = fuse(model, raw_layers, patches, supports, valid)
    with torch.autocast(device_type=mixed.device.type, enabled=False):
        moments = mixed.detach().float()
        neck = F.batch_norm(group_mixed.float(), moments.mean(0), moments.var(0, unbiased=False),
                            model.bottleneck.weight, model.bottleneck.bias,
                            training=False, momentum=0., eps=model.bottleneck.eps)
    scores = model.classifier(model.dropout(neck))
    group_labels = labels[anchors]
    with torch.autocast(device_type=unit.device.type, enabled=False):
        group_id = F.cross_entropy(scores.float(), group_labels)
        target = F.normalize(neck.float(), dim=1)
        active = mapping >= 0
        cosine = (unit[active].float()*target.detach()[mapping[active]]).sum(1)
        ip = (1.-cosine).mean()
        mean_raw = member_mean(mixed.detach().float(), supports, valid)
        mean_neck = F.batch_norm(mean_raw, moments.mean(0), moments.var(0, unbiased=False),
                                model.bottleneck.weight.detach(), model.bottleneck.bias.detach(),
                                training=False, momentum=0., eps=model.bottleneck.eps)
        mean_cosine = (target.detach()*F.normalize(mean_neck, dim=1)).sum(1)
    settings = model.mvi_cross_settings
    result.update(group_id=group_id, ip=ip,
                  weighted=factor*(settings.GROUP_ID_WEIGHT*model.mvi_cross_id_weight*group_id+settings.IP_WEIGHT*ip))
    stats.update(group_correct=int((scores.detach().argmax(1)==group_labels).sum()),
                 ip_cosine_sum=float(cosine.detach().sum()), mean_cosine_sum=float(mean_cosine.sum()))
    return result
