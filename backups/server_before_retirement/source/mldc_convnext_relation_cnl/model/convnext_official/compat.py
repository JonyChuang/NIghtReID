"""Small timm utility equivalents; no change to official ConvNeXt state keys."""
import torch
from torch import nn

trunc_normal_ = nn.init.trunc_normal_


def register_model(function):
    # The experiment constructs the class directly; a global registry is unused.
    return function


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1. - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep) * random.floor()
