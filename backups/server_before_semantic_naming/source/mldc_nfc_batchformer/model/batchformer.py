"""Shared Transformer encoder applied separately to each anchor neighborhood."""
import torch
from torch import nn


class BatchFormer(nn.Module):
    def __init__(self, dim=768, heads=4, dropout=.5):
        super().__init__()
        self.encoder=nn.TransformerEncoderLayer(d_model=dim,nhead=heads,
            dim_feedforward=dim,dropout=dropout,activation='relu',
            batch_first=False,norm_first=False)

    def forward(self, anchors, context, padding):
        if not self.training:
            return anchors
        sequence=torch.cat((anchors[:,None,:],context.detach().to(anchors.dtype)),dim=1)
        mask=torch.cat((torch.zeros(len(anchors),1,device=anchors.device,dtype=torch.bool),padding),dim=1)
        return self.encoder(sequence.transpose(0,1),src_key_padding_mask=mask)[0]
