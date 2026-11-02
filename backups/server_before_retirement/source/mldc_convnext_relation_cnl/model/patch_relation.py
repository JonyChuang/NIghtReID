"""Image-wise off-diagonal cosine-Gram matching, inspired by structural KD."""
import torch
from torch.nn import functional as F


def patch_relation_loss(student_tokens, teacher_tokens):
    """[B,N,Cs] and [B,N,Ct], with corresponding spatial patch positions.

    Both channels are independently L2-normalized. MSE is averaged over all
    ordered off-diagonal patch pairs and images. CLS is excluded by the caller.
    This is our explicit migration objective, not a claimed full FSKD reproduction.
    """
    if (student_tokens.ndim != 3 or teacher_tokens.ndim != 3
            or student_tokens.shape[:2] != teacher_tokens.shape[:2]
            or student_tokens.shape[1] < 2):
        raise ValueError('Corresponding image/patch dimensions must match; channels may differ')
    with torch.autocast(device_type=student_tokens.device.type, enabled=False):
        student = F.normalize(student_tokens.float(), dim=-1, eps=1e-6)
        teacher = F.normalize(teacher_tokens.detach().float(), dim=-1, eps=1e-6)
        student_relation = student @ student.transpose(1, 2)
        teacher_relation = teacher @ teacher.transpose(1, 2)
        mask = ~torch.eye(student.shape[1], device=student.device, dtype=torch.bool)
        difference = (student_relation - teacher_relation)[:, mask]
        loss = difference.square().mean()
        stats = dict(images=student.shape[0], patches=student.shape[1],
            ordered_pairs_per_image=int(mask.sum()),
            teacher_relation_std=float(teacher_relation[:, mask].std(unbiased=False)),
            student_relation_std=float(student_relation.detach()[:, mask].std(unbiased=False)),
            relation_mae=float(difference.detach().abs().mean()))
    return loss, stats
