"""Local CNN texture matches supervise relations of final ReID descriptors."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def local_pair_scores(tokens,confidence,grid,topk=4,band_rows=2,min_patch_conf=.5,min_matches=2):
    batch,patches,_=tokens.shape
    if patches!=grid[0]*grid[1] or confidence.shape!=(batch,patches):
        raise ValueError('Texture descriptors and patch-confidence grids must align')
    if topk<1 or min_matches<1 or min_matches>topk or band_rows<1:
        raise ValueError('Invalid local matching configuration')
    tokens=F.normalize(tokens.float(),dim=-1)
    cosine=torch.einsum('ipc,jqc->ijpq',tokens,tokens)
    row=torch.arange(patches,device=tokens.device)//grid[1]
    allowed=(row[:,None]//band_rows)==(row[None,:]//band_rows)
    candidates=cosine.masked_fill(~allowed,-torch.inf)
    forward=candidates.argmax(-1); reverse=candidates.argmax(-2)
    mutual=reverse.gather(-1,forward)==torch.arange(patches,device=tokens.device)
    quality=confidence[:,None,:]*confidence[None,:,:].expand(batch,batch,patches).gather(-1,forward)
    valid=mutual&(quality>=min_patch_conf**2)
    # Require each endpoint to pass its own confidence threshold as well.
    valid&=(confidence[:,None,:]>=min_patch_conf)
    valid&=(confidence[None,:,:].expand(batch,batch,patches).gather(-1,forward)>=min_patch_conf)
    values=cosine.gather(-1,forward[...,None]).squeeze(-1)
    k=min(topk,patches)
    ranked=values.masked_fill(~valid,-torch.inf).topk(k,dim=-1).indices
    selected=valid.gather(-1,ranked); weight=quality.gather(-1,ranked)*selected
    selected_values=values.gather(-1,ranked)
    score=(selected_values*weight).sum(-1)/weight.sum(-1).clamp_min(1e-8)
    pair_conf=weight.sum(-1)/k
    enough=selected.sum(-1)>=min_matches
    score=score.masked_fill(~enough,0); pair_conf=pair_conf.masked_fill(~enough,0)
    return (score+score.T)*.5,(pair_conf+pair_conf.T)*.5


def relative_distillation(unit,teacher_score,pair_conf,labels,cameras,indices,temperature=.2,
                          gap=.05,hard_negatives=8,min_pair_conf=.25):
    batch=len(unit)
    if (unit.ndim!=2 or unit.shape[1]!=768 or labels.shape!=(batch,) or cameras.shape!=(batch,)
            or indices.shape!=(batch,) or teacher_score.shape!=(batch,batch) or pair_conf.shape!=(batch,batch)):
        raise ValueError('Final BN-after 768D descriptors and image metadata must align')
    if temperature<=0 or not 0<=gap<1 or hard_negatives<1:
        raise ValueError('Invalid relative-distillation settings')
    student=F.normalize(unit.float(),dim=1); similarity=student@student.T
    teacher_score=teacher_score.detach().float(); pair_conf=pair_conf.detach().float()
    distinct=indices[:,None]!=indices[None,:]
    cross=cameras[:,None]!=cameras[None,:]
    same=labels[:,None]==labels[None,:]
    positive=same&cross&distinct; negative=(~same)&cross&distinct
    # Keep the ordinary PK sampler; mine within-batch cross-camera hard negatives.
    ranked=similarity.detach().masked_fill(~negative,-torch.inf).topk(min(hard_negatives,batch),dim=1).indices
    hard=torch.zeros_like(negative); hard.scatter_(1,ranked,True); hard&=negative
    eligible=positive[:,:,None]&hard[:,None,:]
    delta_teacher=teacher_score[:,:,None]-teacher_score[:,None,:]
    delta_student=similarity[:,:,None]-similarity[:,None,:]
    confidence=(pair_conf[:,:,None]*pair_conf[:,None,:]).clamp_min(0).sqrt()
    reliable=eligible&(delta_teacher>gap)&(pair_conf[:,:,None]>=min_pair_conf)&(pair_conf[:,None,:]>=min_pair_conf)
    weight=confidence*((delta_teacher-gap)/(1-gap)).clamp(0,1)*reliable
    if reliable.any():
        logits=delta_student[reliable]/temperature
        target=torch.sigmoid(delta_teacher[reliable]/temperature)
        # Binary KL between the teacher and student's positive-vs-negative preferences.
        entropy=-(target*target.clamp_min(1e-8).log()+(1-target)*(1-target).clamp_min(1e-8).log())
        divergence=(F.binary_cross_entropy_with_logits(logits,target,reduction='none')-entropy).clamp_min(0)
        loss=(divergence*weight[reliable]).sum()/weight[reliable].sum().clamp_min(1e-8)
    else: loss=unit.sum()*0.
    stats=dict(anchors=batch,positive_pairs=int(positive.sum()),eligible_triplets=int(eligible.sum()),
        reliable_triplets=int(reliable.sum()),active_anchors=int(reliable.flatten(1).any(1).sum()),
        confident_pairs=int((pair_conf>=min_pair_conf).sum()),
        candidate_teacher_gap_mean=float(delta_teacher[eligible].mean()) if eligible.any() else 0.,
        candidate_teacher_gap_max=float(delta_teacher[eligible].max()) if eligible.any() else 0.,
        mean_teacher_gap=float(delta_teacher[reliable].mean()) if reliable.any() else 0.,
        mean_pair_confidence=float(confidence[reliable].mean()) if reliable.any() else 0.)
    return loss,stats
