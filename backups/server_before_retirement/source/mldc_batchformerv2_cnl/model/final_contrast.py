"""Final-feature cross-model InfoNCE with training identity-aware negatives.

Uses the final-feature contrastive supervision of FBT (ICCV 2025), with small
dimension-matching projections as in CRD (ICLR 2020). This is not CRD's NCE loss
or the full fusion architecture of FBT. No memory bank or spatial matching.
"""
import torch
from torch import nn
from torch.nn import functional as F


class FinalContrast(nn.Module):
    def __init__(self,dim=128,temperature=.1,cross_camera_positives=False):
        super().__init__()
        if temperature<=0: raise ValueError('InfoNCE temperature must be positive')
        self.student=nn.Linear(768,dim,bias=False)
        self.teacher=nn.Linear(256,dim,bias=False)
        self.temperature=float(temperature)
        self.cross_camera_positives=bool(cross_camera_positives)

    def forward(self,student,teacher,labels,cameras,indices):
        if (student.ndim!=2 or student.shape[1]!=768 or teacher.shape!=(len(student),256)
                or labels.shape!=(len(student),) or cameras.shape!=labels.shape or indices.shape!=labels.shape):
            raise ValueError('Align final student/teacher embeddings and training metadata')
        q=F.normalize(self.student(student.float()),dim=1)
        k=F.normalize(self.teacher(teacher.detach().float()),dim=1)
        logits=(q@k.T)/self.temperature
        same=labels[:,None]==labels[None,:]
        positive=indices[:,None]==indices[None,:]
        if self.cross_camera_positives:
            positive|=same&(cameras[:,None]!=cameras[None,:])
        negative=~same
        active=positive.any(1)&negative.any(1)
        denominator=positive|negative
        logp=logits-torch.logsumexp(logits.masked_fill(~denominator,-torch.inf),dim=1,keepdim=True)
        if active.any():
            loss=-(logp.masked_fill(~positive,0.).sum(1)[active]/positive.sum(1)[active]).mean()
        else: loss=student.sum()*0
        with torch.no_grad():
            similarity=q@k.T
            stats=dict(anchors=len(student),active_anchors=int(active.sum()),positive_pairs=int(positive.sum()),
                negative_pairs=int(negative.sum()),same_identity_negatives=int((negative&same).sum()),
                positive_cosine=float(similarity[positive].mean()),
                negative_cosine=float(similarity[negative].mean()) if negative.any() else 0.,
                teacher_embedding_variance=float(teacher.float().var(0,unbiased=False).mean()))
        return loss,stats
