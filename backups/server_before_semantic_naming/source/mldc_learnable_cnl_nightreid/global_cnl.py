"""Single-space, supervised cross-camera neighbor loss for the fused CLS.

Removing the shallow branch removes the original two-space intersection. This
adaptation mines neighbors in the global space only, using training labels to
exclude wrong identities and same-camera positives.
"""
import torch
from torch import nn
from torch.nn import functional as F


class GlobalCNL(nn.Module):
    def __init__(self,labels,cameras,dim=768,gamma=.9,temperature=.05,momentum=.2):
        super().__init__()
        if not 0<gamma<=1 or temperature<=0 or not 0<=momentum<1:
            raise ValueError('Invalid global CNL gamma, temperature or momentum')
        labels=torch.as_tensor(labels,dtype=torch.long);cameras=torch.as_tensor(cameras,dtype=torch.long)
        if labels.ndim!=1 or cameras.shape!=labels.shape or not labels.numel():
            raise ValueError('CNL requires aligned, nonempty training labels and cameras')
        self.register_buffer('labels',labels);self.register_buffer('cameras',cameras)
        self.register_buffer('features',torch.zeros(len(labels),dim))
        self.register_buffer('valid',torch.zeros(len(labels),dtype=torch.bool))
        self.gamma=float(gamma);self.temperature=float(temperature);self.momentum=float(momentum)

    def validate_batch(self,features,indices,labels,cameras):
        if (features.ndim!=2 or features.shape!=(len(indices),self.features.shape[1]) or
                indices.ndim!=1 or labels.shape!=indices.shape or cameras.shape!=indices.shape):
            raise ValueError('CNL features and training-image metadata must align')
        if not indices.numel() or indices.min()<0 or indices.max()>=len(self.labels):
            raise ValueError('CNL index outside the training memory')
        if not torch.equal(labels,self.labels[indices]) or not torch.equal(cameras,self.cameras[indices]):
            raise ValueError('CNL image indices do not match training identity/camera labels')

    @torch.no_grad()
    def coalesce(self,features,indices):
        # PK replacement can give several augmentations of one image. One slot
        # receives their detached mean direction, independent of their order.
        unique,inverse=indices.unique(sorted=True,return_inverse=True)
        summed=torch.zeros(len(unique),features.shape[1],device=features.device,dtype=torch.float32)
        summed.index_add_(0,inverse,features.detach().float())
        return unique,F.normalize(summed,dim=1)

    def forward(self,features,indices,labels,cameras):
        self.validate_batch(features,indices,labels,cameras)
        with torch.autocast(device_type=features.device.type,enabled=False):
            queries=F.normalize(features.float(),dim=1)
            unique,current=self.coalesce(queries,indices)
            old=self.valid.clone();old[unique]=False
            old_indices=old.nonzero(as_tuple=True)[0]
            candidates=torch.cat((old_indices,unique))
            keys=torch.cat((self.features[old_indices],current))
            similarity=queries@keys.T
            with torch.no_grad():
                eligible=indices[:,None]!=candidates[None,:]
                same=labels[:,None]==self.labels[candidates][None,:]
                cross=cameras[:,None]!=self.cameras[candidates][None,:]
                positives=same&cross&eligible
                maximum=similarity.masked_fill(~eligible,-torch.inf).amax(1,keepdim=True)
                neighbors=(similarity>self.gamma*maximum)&eligible
                selected=neighbors&positives
                # All other identities are negatives. Unselected same-ID images
                # remain excluded, including positives below the threshold.
                negatives=(~same)&eligible
                denominator=negatives|selected
                active=selected.any(1)&negatives.any(1)
            if active.any():
                logits=similarity[active]/self.temperature
                pos=selected[active]
                logp=logits-torch.logsumexp(logits.masked_fill(~denominator[active],-torch.inf),dim=1,keepdim=True)
                loss=-(logp.masked_fill(~pos,0.).sum(1)/pos.sum(1)).mean()
            else:
                loss=similarity.sum()*0.
            stats=dict(anchors=len(indices),active_anchors=int(active.sum()),
                empty_positive_sets=int((~selected.any(1)).sum()),
                anchors_with_crosscam_positive=int(positives.any(1).sum()),
                selected_positives=int(selected.sum()),available_positives=int(positives.sum()),
                candidate_images=len(candidates),memory_images=int(self.valid.sum()))
            return loss,stats

    @torch.no_grad()
    def update(self,features,indices,labels,cameras):
        """Called after a successful optimizer update only; never by forward/eval."""
        self.validate_batch(features,indices,labels,cameras)
        unique,current=self.coalesce(F.normalize(features.float(),dim=1),indices)
        mixed=F.normalize(self.momentum*self.features[unique]+(1-self.momentum)*current,dim=1)
        self.features[unique]=torch.where(self.valid[unique,None],mixed,current)
        self.valid[unique]=True

