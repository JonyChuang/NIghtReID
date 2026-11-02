"""Apply the fused query's BN transform to detached layer references once."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def shared_bn_references(layers, fused_input, bottleneck):
    """Use the current fused batch's statistics, or running statistics in eval.

    This function never calls bottleneck.forward and never updates BN buffers.
    Affine parameters, layers and statistics do not receive reference gradients.
    The returned references are FP32 unit vectors [batch, layers, embedding].
    """
    with torch.autocast(device_type=layers.device.type, enabled=False):
        source = fused_input.detach().float()
        if bottleneck.training or bottleneck.running_mean is None:
            variance, mean = torch.var_mean(source, dim=0, unbiased=False)
        else:
            mean = bottleneck.running_mean.detach().float()
            variance = bottleneck.running_var.detach().float()
        result = (layers.detach().float() - mean[None, None]) * torch.rsqrt(
            variance[None, None] + bottleneck.eps)
        if bottleneck.weight is not None:
            result = result * bottleneck.weight.detach().float()[None, None]
        if bottleneck.bias is not None:
            result = result + bottleneck.bias.detach().float()[None, None]
        return F.normalize(result, dim=-1)
