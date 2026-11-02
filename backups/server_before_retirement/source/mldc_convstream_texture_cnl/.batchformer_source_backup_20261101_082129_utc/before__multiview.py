"""Training-only multi-image fusion using the existing ViT tail and readout."""
import torch
import torch.nn.functional as F


def activation_factor(epoch, warmup_epochs, ramp_epochs):
    if int(epoch) <= warmup_epochs:
        return 0.
    return min(1., (int(epoch) - warmup_epochs) / max(ramp_epochs, 1))


def make_image_pairs(labels, cameras, image_indices, require_cross_camera=True):
    """Disjoint same-ID pairs; repeated augmentations are one source image.

    Pair the two largest remaining camera buckets first. The returned mapping
    also assigns repeated augmentations to their source image's group.
    """
    label_list = labels.detach().cpu().tolist()
    camera_list = cameras.detach().cpu().tolist()
    index_list = image_indices.detach().cpu().tolist()
    by_identity, seen = {}, {}
    for position, (label, camera, image_index) in enumerate(
            zip(label_list, camera_list, index_list)):
        if image_index in seen:
            prior = seen[image_index]
            if (label, camera) != prior[:2]:
                raise ValueError('One image index has inconsistent identity/camera metadata')
            continue
        seen[image_index] = (label, camera, position)
        by_identity.setdefault(label, {}).setdefault(camera, []).append(position)
    pairs, cross_camera = [], 0
    for buckets in by_identity.values():
        while sum(map(len, buckets.values())) >= 2:
            available = sorted((key for key in buckets if buckets[key]),
                               key=lambda key: -len(buckets[key]))
            if len(available) >= 2:
                pair = (buckets[available[0]].pop(0), buckets[available[1]].pop(0))
                cross_camera += 1
            elif not require_cross_camera:
                bucket = buckets[available[0]]
                pair = (bucket.pop(0), bucket.pop(0))
            else:
                break
            pairs.append(pair)
    groups = torch.tensor(pairs, dtype=torch.long, device=labels.device).reshape(-1, 2)
    source_groups = {}
    for group, pair in enumerate(pairs):
        for position in pair:
            source_groups[index_list[position]] = group
    mapping = torch.tensor([source_groups.get(index, -1) for index in index_list],
                           dtype=torch.long, device=labels.device)
    return groups, mapping, dict(unique_images=len(seen), groups=len(pairs),
                                crosscam_groups=cross_camera,
                                active_anchors=int((mapping >= 0).sum().item()))


def fuse_with_shared_tail(backbone, split_tokens, groups, split_layer):
    """Mean input CLS + concatenated patches, then the original tail blocks."""
    group_count, group_size = groups.shape
    members = split_tokens.index_select(0, groups.reshape(-1)).reshape(
        group_count, group_size, split_tokens.shape[1], split_tokens.shape[2])
    group_cls = members[:, :, :1].mean(dim=1)
    patches = members[:, :, 1:].reshape(group_count, -1, split_tokens.shape[2])
    # Reuse positional information already present in each source token.
    tokens = torch.cat((group_cls, patches), dim=1)
    tail_cls = []
    for block in backbone.blocks[split_layer:]:
        tokens = block(tokens)
        tail_cls.append(tokens[:, 0])
    # Existing MLDC expects all depth slots, but selects only CLS9-12.
    unused = tokens.new_zeros((split_layer, group_count, tokens.shape[-1]))
    return torch.cat((unused, torch.stack(tail_cls, dim=0)), dim=0)


def shared_batch_neck(neck, group_features, single_features):
    """Reuse detached single-image batch moments; never update BN state."""
    with torch.autocast(device_type=group_features.device.type, enabled=False):
        moments = single_features.detach().float()
        mean = moments.mean(dim=0)
        variance = moments.var(dim=0, unbiased=False)
        return F.batch_norm(group_features.float(), mean, variance,
                            neck.weight, neck.bias, training=False,
                            momentum=0., eps=neck.eps)


def multiview_terms(model, split_tokens, groups, mapping, stats,
                    mixed, unit, labels, factor):
    zero = unit.sum() * 0.
    result = dict(group_id=zero, ip=zero, weighted=zero, factor=factor,
                  stats=dict(anchors=len(labels), **stats))
    if factor == 0. or groups.numel() == 0:
        return result
    group_layers = fuse_with_shared_tail(model.base, split_tokens, groups,
                                        model.mvi_settings.SPLIT_LAYER)
    group_mixed = model.mldc(group_layers)
    group_neck = shared_batch_neck(model.bottleneck, group_mixed, mixed)
    group_scores = model.classifier(model.dropout(group_neck))
    group_labels = labels.index_select(0, groups[:, 0])
    with torch.autocast(device_type=unit.device.type, enabled=False):
        group_id = F.cross_entropy(group_scores.float(), group_labels)
        group_unit = F.normalize(group_neck.float(), dim=1)
        active = mapping >= 0
        # Group ID learns through the full group path; IP has no group-side edge.
        target = group_unit.detach().index_select(0, mapping[active])
        cosine = (unit[active].float() * target).sum(dim=1)
        ip = (1. - cosine).mean()
        baseline_mean = F.normalize(unit.detach()[groups].mean(dim=1), dim=1)
        mean_cosine = (group_unit.detach() * baseline_mean).sum(dim=1)
    weighted = factor * (model.mvi_settings.GROUP_ID_WEIGHT * model.mvi_id_weight * group_id
                         + model.mvi_settings.IP_WEIGHT * ip)
    result.update(group_id=group_id, ip=ip, weighted=weighted)
    result['stats'].update(
        group_correct=int((group_scores.detach().argmax(1) == group_labels).sum().item()),
        ip_cosine_sum=float(cosine.detach().sum().item()),
        group_mean_cosine_sum=float(mean_cosine.sum().item()),
        group_tokens=1 + groups.shape[1] * (split_tokens.shape[1] - 1))
    return result
