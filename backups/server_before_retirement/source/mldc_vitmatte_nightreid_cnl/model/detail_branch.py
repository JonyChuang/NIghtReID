"""ViTMatte's MIT-licensed detail stream/fusion, adapted to identity readout.

Upstream primitives are retained verbatim in vitmatte_official.py. See
licenses/ViTMatte-MIT.txt and provenance.json for attribution and adaptations.
"""
import math
import torch
from torch import nn
from .vitmatte_official import ConvStream, Fusion_Block


class ViTMatteDetails(nn.Module):
    def __init__(self, dim=768, output_dim=256, pooling='guided'):
        super().__init__()
        if pooling not in ('guided','mean'):
            raise ValueError('Detail pooling must be guided or mean')
        self.pooling=pooling
        self.spatial_norm=nn.LayerNorm(dim)
        self.convstream=ConvStream(in_chans=3)
        # Identical channels/order to upstream Detail_Capture, with RGB input.
        self.fusion_blks=nn.ModuleList([
            Fusion_Block(dim+192,256),Fusion_Block(256+96,128),
            Fusion_Block(128+48,64),Fusion_Block(64+3,32)])
        self.token_norm=nn.LayerNorm(32)
        self.query_norm=nn.LayerNorm(dim)
        self.query=nn.Linear(dim,32,bias=False)
        self.project=nn.Linear(32,output_dim)
        self.output_norm=nn.LayerNorm(output_dim)
        if pooling=='mean':
            self.query_norm.requires_grad_(False)
            self.query.requires_grad_(False)

    def capture(self, spatial, images, return_pyramid=False):
        details=self.convstream(images)
        pyramid={}
        for i,block in enumerate(self.fusion_blks):
            detail=details['D'+str(3-i)]
            spatial=block(spatial,detail)
            pyramid['F'+str(3-i)]=spatial
        return (spatial,details,pyramid) if return_pyramid else spatial

    def forward(self, patches, images, global_feature, return_maps=False):
        height,width=images.shape[-2:]
        if height%16 or width%16 or patches.shape[1]!=(height//16)*(width//16):
            raise ValueError('ViTMatte transfer requires a stride-16 grid and input dimensions divisible by 16')
        spatial=self.spatial_norm(patches).transpose(1,2).reshape(len(images),-1,height//16,width//16)
        result=self.capture(spatial,images,return_pyramid=return_maps)
        feature_map=result[0] if return_maps else result
        tokens=self.token_norm(feature_map.flatten(2).transpose(1,2))
        if self.pooling=='guided':
            query=self.query(self.query_norm(global_feature))
            # Pool in FP32 under AMP; the detail stream remains differentiable.
            attention=torch.softmax((tokens.float()*query.float().unsqueeze(1)).sum(-1)/math.sqrt(32),dim=1)
            pooled=(tokens.float()*attention.unsqueeze(-1)).sum(1)
        else:
            attention=tokens.new_full(tokens.shape[:2],1/tokens.shape[1],dtype=torch.float32)
            pooled=tokens.float().mean(1)
        descriptor=self.output_norm(self.project(pooled))
        if return_maps:
            return descriptor,dict(attention=attention.reshape(len(images),height,width),
                detail_shapes={k:list(v.shape) for k,v in result[1].items()},
                fusion_shapes={k:list(v.shape) for k,v in result[2].items()})
        return descriptor
