"""Training-only batch reciprocal-neighbor centers for single-image contrastive learning."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class BatchNFCCenterLoss(nn.Module):
    """No teacher, cache or memory. Only anchor features receive gradients.

    The top2 reciprocal graph is label/camera blind. Labels/cameras select
    cross-camera positive owners and different-ID negative keys afterwards.
    Graph members are NOT filtered by identity. A center excludes every copy
    of the anchor image and must retain at least two distinct support images.
    """
    def __init__(self,temperature=.05,weight=.1,warmup_epochs=20,ramp_epochs=5):
        super().__init__()
        if (not math.isfinite(temperature) or temperature<=0 or not math.isfinite(weight)
                or weight<0 or warmup_epochs<0 or ramp_epochs<0):
            raise ValueError('Invalid batch NFC center loss settings')
        self.temperature=float(temperature);self.weight=float(weight)
        self.warmup_epochs=int(warmup_epochs);self.ramp_epochs=int(ramp_epochs)

    def current_weight(self,epoch):
        if epoch<=self.warmup_epochs:return 0.
        fraction=min(1.,(epoch-self.warmup_epochs)/self.ramp_epochs) if self.ramp_epochs else 1.
        return self.weight*fraction

    @torch.no_grad()
    def _graph(self,queries,indices,labels,cameras):
        unique,inverse=indices.unique(sorted=True,return_inverse=True)
        # Each original image owns exactly one graph vertex, even under PK replacement.
        first=(indices[:,None]==unique[None,:]).long().argmax(0)
        identity=labels[first];camera=cameras[first]
        if not torch.equal(identity[inverse],labels) or not torch.equal(camera[inverse],cameras):
            raise ValueError('Repeated image indices must have consistent identity and camera')
        keys=torch.zeros(len(unique),queries.shape[1],device=queries.device,dtype=torch.float32)
        keys.index_add_(0,inverse,queries.detach())
        keys=F.normalize(keys,dim=1)
        norms=keys.square().sum(1)
        distance=norms[:,None]+norms[None,:]-2*keys@keys.T
        distance.fill_diagonal_(1000)
        nearest=distance.topk(2,largest=False).indices
        mutual=(nearest[nearest]==torch.arange(len(unique),device=keys.device)[:,None,None]).any(2)
        support=torch.eye(len(unique),device=keys.device,dtype=torch.bool)
        support.scatter_(1,nearest,mutual)
        return unique,inverse,identity,camera,keys,nearest,mutual,support

    def forward(self,features,indices,labels,cameras,epoch):
        if (features.ndim!=2 or indices.ndim!=1 or len(features)!=len(indices)
                or labels.shape!=indices.shape or cameras.shape!=indices.shape):
            raise ValueError('NFC batch features and image metadata must align')
        with torch.autocast(device_type=features.device.type,enabled=False):
            queries=F.normalize(features.float(),dim=1)
            zero=queries.sum()*0.
            stats=dict(anchors=len(indices),unique_images=int(indices.unique().numel()),
                graph_edges=0,graph_same_id_edges=0,graph_cross_camera_edges=0,
                graph_active_images=0,crosscam_positive_pairs=0,positive_centers=0,
                active_anchors=0,support_images=0,margin_sum=0.,graph_computed=0)
            if self.current_weight(epoch)==0 or stats['unique_images']<3:
                return zero,stats
            with torch.no_grad():
                unique,inverse,identity,camera,keys,nearest,mutual,support=self._graph(queries,indices,labels,cameras)
                same=labels[:,None]==identity[None,:]
                positive=same&(cameras[:,None]!=camera[None,:])&(indices[:,None]!=unique[None,:])
                negative=~same
                # Removing the anchor vertex also removes every augmented copy of it.
                counts=support.sum(1)[None,:]-support[:,inverse].T.long()
                selected=positive&(counts>=2)&negative.any(1)[:,None]
                anchor,owner=selected.nonzero(as_tuple=True)
                stats.update(graph_edges=int(mutual.sum()),
                    graph_same_id_edges=int(((identity[nearest]==identity[:,None])&mutual).sum()),
                    graph_cross_camera_edges=int(((camera[nearest]!=camera[:,None])&mutual).sum()),
                    graph_active_images=int(mutual.any(1).sum()),crosscam_positive_pairs=int(positive.sum()),
                    positive_centers=len(anchor),active_anchors=int(anchor.unique().numel()),graph_computed=1)
                if not len(anchor):return zero,stats
                coefficients=support[owner]&(torch.arange(len(unique),device=keys.device)[None,:]!=inverse[anchor,None])
                centers=F.normalize(coefficients.float()@keys,dim=1).detach()
                stats['support_images']=int(coefficients.sum())
            similarity=queries@keys.T
            negative_lse=(similarity/self.temperature).masked_fill(~negative,-torch.inf).logsumexp(1)
            positive_similarity=(queries[anchor]*centers).sum(1)
            loss=F.softplus(negative_lse[anchor]-positive_similarity/self.temperature).mean()
            with torch.no_grad():
                hardest_negative=similarity.detach().masked_fill(~negative,-torch.inf).max(1).values
                stats['margin_sum']=float((positive_similarity.detach()-hardest_negative[anchor]).sum())
            return loss,stats
