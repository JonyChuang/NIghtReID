"""FFVT selection and fusion in the existing experiment package.

MAWS: https://github.com/Markin-Wang/FFVT . Normalize the CLS row and
column per head, average heads, then multiply. CLS is excluded from top-k.
"""
import copy
import torch
import torch.nn as nn


@torch.no_grad()
def mutual_attention_scores(logits):
    """Scores [B,N] before dropout; CLS column softmax is over query positions."""
    cls_row = logits[:, :, 0, :].float().softmax(dim=-1).mean(dim=1)
    cls_column = logits[:, :, :, 0].float().softmax(dim=-1).mean(dim=1)
    return cls_row * cls_column


def select_patch_tokens(tokens, scores, count):
    """Gather post-block tokens; discrete indices, differentiable token values."""
    if not 1 <= count < tokens.shape[1]:
        raise ValueError('FFVT_TOKENS_PER_LAYER must be within the patch count')
    indices = scores[:, 1:].topk(count, dim=1, sorted=True).indices + 1
    selected = tokens.gather(1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
    return selected, indices


class CrossLayerFusion(nn.Module):
    """An independently trainable copy of the pretrained final block and LN."""
    def __init__(self, final_block, final_norm):
        super().__init__()
        self.block = copy.deepcopy(final_block)
        self.norm = copy.deepcopy(final_norm)

    def forward(self, cls_token, selected_patches):
        sequence = torch.cat((cls_token, selected_patches), dim=1)
        return self.norm(self.block(sequence))[:, 0]
