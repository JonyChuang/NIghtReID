#!/usr/bin/env python3
"""Analyze per-head Block12 Q-Norm from a frozen TransReID-SSL teacher.

This script is intentionally analysis-only: it never trains, updates, or saves
model parameters. It uses a fixed list of 20 validation-set indices and stores
the uninterpolated per-head arrays alongside the rendered figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# Fixed once for this analysis. Do not replace this with random sampling.
FIXED_SAMPLE_INDICES: Tuple[int, ...] = (
    245, 509, 782, 830, 923,
    3834, 3917, 4472, 4756, 5435,
    6022, 7195, 7382, 8004, 8337,
    9615, 10212, 11116, 12994, 15234,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config_file",
        default="experiment_variants/qnorm_context_raw/config.yml",
        help="Night600 model/data configuration.",
    )
    parser.add_argument(
        "--teacher_checkpoint",
        default=(
            "../../log/transreid/night600/vit_base_ics_teacher/"
            "transformer_teacher_120.pth"
        ),
        help="Frozen teacher checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        default="../../log/transreid/night600/teacher_per_head_qnorm_analysis",
        help="New analysis-only output directory.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a concrete device such as cuda:0.",
    )
    parser.add_argument(
        "--start_sample",
        type=int,
        default=0,
        help="Zero-based offset in the fixed 20-sample list.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of consecutive entries from the fixed list to process.",
    )
    parser.add_argument(
        "--rerender_saved",
        action="store_true",
        help=(
            "Overwrite visualizations from the saved raw qnorm_analysis.npz; "
            "the model is not loaded and numeric artifacts are not rewritten."
        ),
    )
    parser.add_argument(
        "--t_mean",
        type=float,
        default=1.0,
        help="Temperature for sigmoid(-z_mean / T_mean).",
    )
    parser.add_argument(
        "--t_std",
        type=float,
        default=1.0,
        help="Temperature for exp(-z_std / T_std).",
    )
    return parser.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def checkpoint_state(path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Teacher checkpoint does not contain a state dict")
    state = {}
    for key, value in checkpoint.items():
        if torch.is_tensor(value):
            clean_key = key[7:] if key.startswith("module.") else key
            state[clean_key] = value
    return state


def load_frozen_teacher(cfg, num_classes: int, camera_num: int, view_num: int,
                        checkpoint_path: str, device: torch.device):
    # The QCAT-Rank wrapper only exposes the already verified Block12 aux; it
    # does not alter the teacher's attention computation or checkpoint weights.
    from model.make_model_qcat_rank import make_model

    model = make_model(
        cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num
    )
    state = checkpoint_state(checkpoint_path)
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = [
        key for key in incompatible.unexpected_keys
        if "qnorm_cross_attn" not in key
    ]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            "Teacher checkpoint is incompatible with the TransReID backbone: "
            "missing={}, unexpected={}".format(
                incompatible.missing_keys, unexpected
            )
        )
    model.to(device).requires_grad_(False).eval()
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Teacher must remain frozen and in eval mode")
    return model


def load_initial_student(cfg, num_classes: int, camera_num: int, view_num: int,
                         device: torch.device):
    """Build the same pretrained, pre-training Student used by QCAT 'Before'."""
    from model.make_model_qcat_rank import make_model

    model = make_model(
        cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num
    )
    model.to(device).requires_grad_(False).eval()
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Initial Student must remain frozen and in eval mode")
    return model


def minmax(values: np.ndarray) -> np.ndarray:
    low = float(values.min())
    high = float(values.max())
    return (values - low) / (high - low + 1e-12)


def compute_statistics(qnorm_heads: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute statistics from an uninterpolated [H, N] Q-Norm tensor."""
    if qnorm_heads.ndim != 2:
        raise ValueError("qnorm_heads must have shape [H, N]")
    heads, patches = qnorm_heads.shape
    if heads < 1 or patches < 1:
        raise ValueError("Q-Norm tensor is empty")

    mean_map = qnorm_heads.mean(dim=0)
    std_map = qnorm_heads.std(dim=0, unbiased=False)
    mean_low = 1.0 - (mean_map - mean_map.min()) / (
        mean_map.max() - mean_map.min() + 1e-12
    )
    std_low = 1.0 - (std_map - std_map.min()) / (
        std_map.max() - std_map.min() + 1e-12
    )
    low_mean_low_std_map = mean_low * std_low

    normalized_heads = F.normalize(qnorm_heads, p=2, dim=-1)
    cosine_matrix = normalized_heads @ normalized_heads.transpose(0, 1)

    top_k = max(1, int(patches * 0.20))
    top_indices = qnorm_heads.topk(top_k, dim=-1).indices
    top_masks = torch.zeros(
        heads, patches, dtype=torch.bool, device=qnorm_heads.device
    )
    top_masks.scatter_(1, top_indices, True)
    intersection = (top_masks[:, None, :] & top_masks[None, :, :]).sum(-1).float()
    union = (top_masks[:, None, :] | top_masks[None, :, :]).sum(-1).float()
    top20_iou_matrix = intersection / union.clamp_min(1.0)

    probability = qnorm_heads / qnorm_heads.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    spatial_entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
    top20_concentration = qnorm_heads.gather(1, top_indices).sum(dim=-1) / (
        qnorm_heads.sum(dim=-1).clamp_min(1e-12)
    )

    return {
        "mean_map": mean_map,
        "std_map": std_map,
        "low_mean_low_std_map": low_mean_low_std_map,
        "cosine_matrix": cosine_matrix,
        "top20_iou_matrix": top20_iou_matrix,
        "mean_qnorm": qnorm_heads.mean(dim=-1),
        "std_qnorm": qnorm_heads.std(dim=-1, unbiased=False),
        "spatial_entropy": spatial_entropy,
        "top20_concentration": top20_concentration,
    }


def compute_consensus_statistics(
    qnorm_heads: torch.Tensor,
    student_cls_attn_heads: torch.Tensor,
    t_mean: float,
    t_std: float,
) -> Dict[str, torch.Tensor]:
    """Compute all consensus values strictly on the raw patch dimension."""
    if qnorm_heads.ndim != 2 or student_cls_attn_heads.ndim != 2:
        raise ValueError("Teacher Q-Norm and Student attention must be [H, N]")
    if qnorm_heads.shape[-1] != student_cls_attn_heads.shape[-1]:
        raise ValueError("Teacher and Student patch counts do not match")
    if t_mean <= 0 or t_std <= 0:
        raise ValueError("T_mean and T_std must both be positive")

    per_head_mean = qnorm_heads.mean(dim=-1, keepdim=True)
    per_head_std = qnorm_heads.std(dim=-1, keepdim=True, unbiased=False)
    z_heads = (qnorm_heads - per_head_mean) / per_head_std.clamp_min(1e-6)
    z_mean = z_heads.mean(dim=0)
    z_std = z_heads.std(dim=0, unbiased=False)
    low_score = torch.sigmoid(-z_mean / t_mean)
    agreement = torch.exp(-z_std / t_std)
    consensus = low_score * agreement

    student_cls_before = student_cls_attn_heads.mean(dim=0)
    student_cls_before = student_cls_before / student_cls_before.sum().clamp_min(1e-12)
    cls_times_consensus = student_cls_before * consensus
    consensus_mass = cls_times_consensus.sum()
    return {
        "z_heads": z_heads,
        "z_mean": z_mean,
        "z_std": z_std,
        "low_score": low_score,
        "agreement": agreement,
        "consensus": consensus,
        "student_cls_attn_heads": student_cls_attn_heads,
        "student_cls_before": student_cls_before,
        "cls_times_consensus": cls_times_consensus,
        "consensus_mass": consensus_mass,
    }


def denormalize(image: torch.Tensor, mean: Sequence[float],
                std: Sequence[float]) -> np.ndarray:
    mean_tensor = torch.tensor(mean, dtype=image.dtype).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=image.dtype).view(3, 1, 1)
    return (
        (image.cpu() * std_tensor + mean_tensor)
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )


def hide_unused(axes: np.ndarray, used: int) -> None:
    for axis in axes.flat[used:]:
        axis.axis("off")


def save_raw_nearest(rgb: np.ndarray, qnorm_maps: np.ndarray,
                     mean_map: np.ndarray, output_path: Path) -> None:
    heads = qnorm_maps.shape[0]
    panels = heads + 2
    columns = 4
    rows = int(math.ceil(panels / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.asarray(axes).reshape(rows, columns)
    axes.flat[0].imshow(rgb)
    axes.flat[0].set_title("Original Image")
    axes.flat[0].axis("off")

    shared_min = float(qnorm_maps.min())
    shared_max = float(qnorm_maps.max())
    map_axes = []
    for panel, (values, title) in enumerate(
        [(mean_map, "Mean Q-Norm")]
        + [(qnorm_maps[head], "Head {:02d}".format(head + 1)) for head in range(heads)],
        start=1,
    ):
        axis = axes.flat[panel]
        image = axis.imshow(
            values, cmap="turbo", vmin=shared_min, vmax=shared_max,
            interpolation="nearest", aspect="auto",
        )
        axis.set_title(title)
        axis.axis("off")
        map_axes.append(axis)
    hide_unused(axes, panels)
    fig.colorbar(image, ax=map_axes, fraction=0.018, pad=0.02)
    fig.suptitle(
        "Raw per-head Block12 Q-Norm — nearest-neighbor patch map "
        "[{:.4f}, {:.4f}]".format(shared_min, shared_max),
        fontsize=14,
    )
    fig.subplots_adjust(top=0.92, wspace=0.10, hspace=0.22)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def interpolate_qnorm_maps(qnorm_heads: torch.Tensor, grid_h: int, grid_w: int,
                           image_h: int, image_w: int) -> np.ndarray:
    """Bilinearly render [H, N] without changing the raw analysis tensor."""
    heads, patches = qnorm_heads.shape
    if grid_h * grid_w != patches:
        raise RuntimeError(
            "Patch grid {}x{} does not match N={}".format(grid_h, grid_w, patches)
        )
    patch_maps = qnorm_heads.reshape(heads, 1, grid_h, grid_w)
    smooth_maps = F.interpolate(
        patch_maps,
        size=(image_h, image_w),
        mode="bilinear",
        align_corners=False,
    )
    return smooth_maps[:, 0].cpu().numpy()


def save_shared_scale(rgb: np.ndarray, smooth_maps: np.ndarray,
                      smooth_mean_map: np.ndarray, raw_min: float,
                      raw_max: float, output_path: Path) -> None:
    heads = smooth_maps.shape[0]
    panels = heads + 2
    columns = 4
    rows = int(math.ceil(panels / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.asarray(axes).reshape(rows, columns)
    axes.flat[0].imshow(rgb)
    axes.flat[0].set_title("Original Image")
    axes.flat[0].axis("off")
    map_axes = []
    for panel, (values, title) in enumerate(
        [(smooth_mean_map, "Mean Q-Norm")]
        + [(smooth_maps[head], "Head {:02d}".format(head + 1))
           for head in range(heads)],
        start=1,
    ):
        axis = axes.flat[panel]
        image = axis.imshow(
            values, cmap="turbo", vmin=raw_min, vmax=raw_max,
            interpolation="nearest", aspect="auto",
        )
        axis.set_title(title)
        axis.axis("off")
        map_axes.append(axis)
    hide_unused(axes, panels)
    fig.colorbar(image, ax=map_axes, fraction=0.018, pad=0.02)
    fig.suptitle(
        "Bilinear per-head Block12 Q-Norm — shared absolute scale "
        "[{:.4f}, {:.4f}]".format(raw_min, raw_max),
        fontsize=14,
    )
    fig.subplots_adjust(top=0.92, wspace=0.10, hspace=0.22)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_individual_norm(rgb: np.ndarray, smooth_maps: np.ndarray,
                         output_path: Path) -> None:
    heads = smooth_maps.shape[0]
    panels = heads + 1
    columns = 4
    rows = int(math.ceil(panels / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.asarray(axes).reshape(rows, columns)
    axes.flat[0].imshow(rgb)
    axes.flat[0].set_title("Original Image")
    axes.flat[0].axis("off")
    for head in range(heads):
        values = smooth_maps[head]
        normalized = minmax(values)
        axis = axes.flat[head + 1]
        axis.imshow(
            normalized, cmap="turbo", vmin=0, vmax=1,
            interpolation="nearest", aspect="auto",
        )
        axis.set_title(
            "Head {:02d}\nsmooth [{:.3f}, {:.3f}]".format(
                head + 1, float(values.min()), float(values.max())
            )
        )
        axis.axis("off")
    hide_unused(axes, panels)
    fig.suptitle(
        "Bilinear per-head Block12 Q-Norm — individual min-max normalization",
        fontsize=14,
    )
    fig.subplots_adjust(top=0.92, wspace=0.10, hspace=0.28)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_overlay_grid(rgb: np.ndarray, smooth_maps: np.ndarray,
                      output_path: Path, shared_scale: bool,
                      raw_min: float, raw_max: float,
                      alpha: float = 0.45) -> None:
    heads = smooth_maps.shape[0]
    panels = heads + 1
    columns = 4
    rows = int(math.ceil(panels / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.asarray(axes).reshape(rows, columns)
    axes.flat[0].imshow(rgb)
    axes.flat[0].set_title("Original Image")
    axes.flat[0].axis("off")
    for head in range(heads):
        values = smooth_maps[head]
        if shared_scale:
            display = values
            vmin, vmax = raw_min, raw_max
        else:
            display = minmax(values)
            vmin, vmax = 0.0, 1.0
        axis = axes.flat[head + 1]
        axis.imshow(rgb)
        axis.imshow(
            display, cmap="turbo", vmin=vmin, vmax=vmax,
            alpha=alpha, interpolation="nearest", aspect="auto",
        )
        axis.set_title("Head {:02d}".format(head + 1))
        axis.axis("off")
    hide_unused(axes, panels)
    scale_name = "shared absolute scale" if shared_scale else "per-head min-max"
    fig.suptitle(
        "Bilinear Q-Norm overlay (alpha={:.2f}) — {}".format(alpha, scale_name),
        fontsize=14,
    )
    fig.subplots_adjust(top=0.92, wspace=0.08, hspace=0.18)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_overlay_shared_with_mean(
    rgb: np.ndarray,
    smooth_maps: np.ndarray,
    smooth_mean_map: np.ndarray,
    output_path: Path,
    vmin: float,
    vmax: float,
    alpha: float = 0.45,
) -> None:
    """Save the single final figure with a shared absolute Q-Norm scale."""
    img_h, img_w = rgb.shape[:2]
    heads = smooth_maps.shape[0]
    panels = heads + 2
    columns = 4
    rows = int(math.ceil(panels / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.asarray(axes).reshape(rows, columns)

    original_axis = axes.flat[0]
    original_axis.imshow(rgb)
    original_axis.set_title("Original Image")
    original_axis.set_box_aspect(img_h / img_w)
    original_axis.axis("off")

    overlays = [(smooth_mean_map, "Mean Q-Norm")]
    overlays.extend(
        (smooth_maps[head], "Head {:02d}".format(head + 1))
        for head in range(heads)
    )
    overlay_axes = []
    for panel, (q_up, title) in enumerate(overlays, start=1):
        axis = axes.flat[panel]
        axis.imshow(rgb)
        heatmap = axis.imshow(
            q_up,
            cmap="jet",
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(title)
        axis.set_box_aspect(img_h / img_w)
        axis.axis("off")
        overlay_axes.append(axis)

    hide_unused(axes, panels)
    fig.colorbar(heatmap, ax=overlay_axes, fraction=0.018, pad=0.02)
    fig.suptitle(
        "Block12 per-head Q-Norm overlay — shared absolute scale "
        "[{:.4f}, {:.4f}], alpha={:.2f}".format(vmin, vmax, alpha),
        fontsize=14,
    )
    fig.subplots_adjust(top=0.92, wspace=0.12, hspace=0.22)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_consensus_overview(
    rgb: np.ndarray,
    qnorm_heads: torch.Tensor,
    consensus_statistics: Dict[str, torch.Tensor],
    grid_h: int,
    grid_w: int,
    output_path: Path,
    t_mean: float,
    t_std: float,
    alpha: float = 0.45,
) -> None:
    """Render the seven requested overlays from raw patch-domain values."""
    img_h, img_w = rgb.shape[:2]

    def upsample(vector: torch.Tensor) -> np.ndarray:
        return interpolate_qnorm_maps(
            vector.float().cpu().unsqueeze(0),
            grid_h,
            grid_w,
            img_h,
            img_w,
        )[0]

    mean_qnorm = qnorm_heads.mean(dim=0)
    z_mean = consensus_statistics["z_mean"]
    z_std = consensus_statistics["z_std"]
    consensus = consensus_statistics["consensus"]
    student_before = consensus_statistics["student_cls_before"]
    cls_times_consensus = consensus_statistics["cls_times_consensus"]
    consensus_mass = float(consensus_statistics["consensus_mass"])

    z_limit = max(float(z_mean.abs().max()), 1e-8)
    student_limit = max(float(student_before.max()), 1e-12)
    panels = (
        (
            "Mean Q-Norm\n[{:.4f}, {:.4f}]".format(
                float(mean_qnorm.min()), float(mean_qnorm.max())
            ),
            upsample(mean_qnorm),
            float(mean_qnorm.min()),
            float(mean_qnorm.max()),
        ),
        (
            "Cross-head Mean Z-Score\n[{:.4f}, {:.4f}]".format(
                float(z_mean.min()), float(z_mean.max())
            ),
            upsample(z_mean),
            -z_limit,
            z_limit,
        ),
        (
            "Cross-head Std\n[{:.4f}, {:.4f}]".format(
                float(z_std.min()), float(z_std.max())
            ),
            upsample(z_std),
            0.0,
            max(float(z_std.max()), 1e-8),
        ),
        (
            "Low-response Consensus\nTmean={:.2f}, Tstd={:.2f}".format(
                t_mean, t_std
            ),
            upsample(consensus),
            0.0,
            1.0,
        ),
        (
            "Student CLS Before\n[0, {:.6f}]".format(student_limit),
            upsample(student_before),
            0.0,
            student_limit,
        ),
        (
            "CLS x Consensus\nConsensusMass={:.6f}".format(consensus_mass),
            upsample(cls_times_consensus),
            0.0,
            student_limit,
        ),
    )

    fig, axes = plt.subplots(2, 4, figsize=(16, 11))
    axes = np.asarray(axes).reshape(2, 4)
    original_axis = axes.flat[0]
    original_axis.imshow(rgb)
    original_axis.set_title("Original Image")
    original_axis.set_box_aspect(img_h / img_w)
    original_axis.axis("off")

    for axis, (title, values, vmin, vmax) in zip(axes.flat[1:], panels):
        axis.imshow(rgb)
        axis.imshow(
            values,
            cmap="jet",
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(title)
        axis.set_box_aspect(img_h / img_w)
        axis.axis("off")

    axes.flat[-1].set_box_aspect(img_h / img_w)
    axes.flat[-1].axis("off")
    fig.suptitle(
        "Teacher per-head Q-Norm consensus and initial Student Block12 CLS attention",
        fontsize=14,
    )
    fig.subplots_adjust(top=0.90, wspace=0.28, hspace=0.24)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def remove_superseded_pngs(sample_dir: Path, keep_name: str) -> None:
    """After a successful render, keep exactly one PNG for this sample."""
    for png_path in sample_dir.glob("*.png"):
        if png_path.name != keep_name:
            png_path.unlink()


def save_summary_maps(mean_map: np.ndarray, std_map: np.ndarray,
                      low_map: np.ndarray, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, values, title, cmap in (
        (axes[0], mean_map, "head_mean_map", "turbo"),
        (axes[1], std_map, "head_std_map", "magma"),
        (axes[2], low_map, "low_mean_low_std_map", "viridis"),
    ):
        image = axis.imshow(values, cmap=cmap, interpolation="nearest", aspect="auto")
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_single_map(values: np.ndarray, output_path: Path, title: str,
                    cmap: str) -> None:
    fig, axis = plt.subplots(1, 1, figsize=(5.2, 4.6))
    image = axis.imshow(values, cmap=cmap, interpolation="nearest", aspect="auto")
    axis.set_title(title)
    axis.axis("off")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_single_matrix(values: np.ndarray, output_path: Path,
                       title: str) -> None:
    heads = values.shape[0]
    labels = [str(index + 1) for index in range(heads)]
    fig, axis = plt.subplots(1, 1, figsize=(7.2, 6.5))
    image = axis.imshow(values, cmap="viridis", vmin=0, vmax=1)
    axis.set_xticks(range(heads), labels)
    axis.set_yticks(range(heads), labels)
    axis.set_xlabel("Head")
    axis.set_ylabel("Head")
    axis.set_title(title)
    for row in range(heads):
        for column in range(heads):
            value = float(values[row, column])
            axis.text(
                column, row, "{:.2f}".format(value),
                ha="center", va="center", fontsize=6,
                color="white" if value < 0.45 else "black",
            )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_matrix_pair(cosine: np.ndarray, iou: np.ndarray,
                     output_path: Path, title_prefix: str = "") -> None:
    heads = cosine.shape[0]
    labels = [str(index + 1) for index in range(heads)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for axis, values, title in (
        (axes[0], cosine, "Q-Norm cosine similarity"),
        (axes[1], iou, "Top20% patch IoU"),
    ):
        image = axis.imshow(values, cmap="viridis", vmin=0, vmax=1)
        axis.set_xticks(range(heads), labels)
        axis.set_yticks(range(heads), labels)
        axis.set_xlabel("Head")
        axis.set_ylabel("Head")
        axis.set_title(title)
        for row in range(heads):
            for column in range(heads):
                value = float(values[row, column])
                axis.text(
                    column, row, "{:.2f}".format(value),
                    ha="center", va="center", fontsize=6,
                    color="white" if value < 0.45 else "black",
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=14)
        fig.subplots_adjust(top=0.88)
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_head_metrics(path: Path, sample_index: int,
                       statistics: Dict[str, np.ndarray]) -> List[Dict[str, object]]:
    rows = []
    heads = len(statistics["mean_qnorm"])
    for head in range(heads):
        rows.append({
            "sample_index": sample_index,
            "head": head + 1,
            "mean_qnorm": float(statistics["mean_qnorm"][head]),
            "std_qnorm": float(statistics["std_qnorm"][head]),
            "spatial_entropy": float(statistics["spatial_entropy"][head]),
            "top20_concentration": float(statistics["top20_concentration"][head]),
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_sample_figures(rgb: np.ndarray, qnorm_heads_tensor: torch.Tensor,
                          grid_h: int, grid_w: int,
                          statistics: Dict[str, np.ndarray],
                          sample_dir: Path, sample_index: int) -> None:
    """Render figures only; every statistic passed here comes from raw [H, N]."""
    heads, patches = qnorm_heads_tensor.shape
    if grid_h * grid_w != patches:
        raise RuntimeError(
            "Patch grid {}x{} does not match N={}".format(grid_h, grid_w, patches)
        )
    image_h, image_w = rgb.shape[:2]
    smooth_maps = interpolate_qnorm_maps(
        qnorm_heads_tensor, grid_h, grid_w, image_h, image_w
    )
    raw_mean_tensor = torch.from_numpy(statistics["mean_map"]).float().unsqueeze(0)
    smooth_mean_map = interpolate_qnorm_maps(
        raw_mean_tensor, grid_h, grid_w, image_h, image_w
    )[0]
    raw_min = float(qnorm_heads_tensor.min())
    raw_max = float(qnorm_heads_tensor.max())
    output_name = "per_head_qnorm_overlay_shared.png"
    save_overlay_shared_with_mean(
        rgb,
        smooth_maps,
        smooth_mean_map,
        sample_dir / output_name,
        vmin=raw_min,
        vmax=raw_max,
        alpha=0.45,
    )
    remove_superseded_pngs(sample_dir, output_name)


def render_consensus_sample(
    rgb: np.ndarray,
    qnorm_heads_tensor: torch.Tensor,
    consensus_statistics: Dict[str, torch.Tensor],
    grid_h: int,
    grid_w: int,
    sample_dir: Path,
    t_mean: float,
    t_std: float,
) -> None:
    output_name = "qnorm_consensus_student_overlay.png"
    save_consensus_overview(
        rgb,
        qnorm_heads_tensor,
        consensus_statistics,
        grid_h,
        grid_w,
        sample_dir / output_name,
        t_mean,
        t_std,
        alpha=0.45,
    )
    remove_superseded_pngs(sample_dir, output_name)


def main() -> None:
    args = parse_args()
    if len(FIXED_SAMPLE_INDICES) != 20 or len(set(FIXED_SAMPLE_INDICES)) != 20:
        raise RuntimeError("FIXED_SAMPLE_INDICES must contain 20 unique entries")
    if args.start_sample < 0 or args.num_samples < 1:
        raise ValueError("start_sample must be >= 0 and num_samples must be >= 1")
    if args.t_mean <= 0 or args.t_std <= 0:
        raise ValueError("t_mean and t_std must both be positive")
    stop_sample = args.start_sample + args.num_samples
    if stop_sample > len(FIXED_SAMPLE_INDICES):
        raise ValueError("Requested fixed-sample slice exceeds the 20-sample list")
    selected_indices = FIXED_SAMPLE_INDICES[args.start_sample:stop_sample]

    from config import cfg as base_cfg
    from datasets import make_dataloader

    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.JPM = False
    cfg.freeze()

    _, _, val_loader, _, num_classes, camera_num, view_num = make_dataloader(cfg)
    dataset = val_loader.dataset
    if max(FIXED_SAMPLE_INDICES) >= len(dataset):
        raise IndexError(
            "Fixed sample index {} exceeds validation-set length {}".format(
                max(FIXED_SAMPLE_INDICES), len(dataset)
            )
        )

    output_dir = Path(args.output_dir)
    if args.rerender_saved:
        if not output_dir.is_dir():
            raise FileNotFoundError(
                "--rerender_saved requires an existing output directory: {}".format(
                    output_dir
                )
            )
        device = resolve_device(args.device)
        initial_student = load_initial_student(
            cfg, num_classes, camera_num, view_num, device
        )
        student_grid_h = int(initial_student.base.patch_embed.num_y)
        student_grid_w = int(initial_student.base.patch_embed.num_x)
        consensus_rows: List[Dict[str, object]] = []
        for global_order, sample_index in enumerate(
            selected_indices, start=args.start_sample + 1
        ):
            image, pid, camid, viewid, image_path = dataset[sample_index]
            sample_dir = output_dir / "sample_{:02d}_idx_{:05d}".format(
                global_order, sample_index
            )
            npz_path = sample_dir / "qnorm_analysis.npz"
            if not npz_path.is_file():
                raise FileNotFoundError(
                    "Saved raw Q-Norm is required for rerender: {}".format(npz_path)
                )
            with np.load(npz_path, allow_pickle=False) as saved:
                if int(saved["sample_index"]) != sample_index:
                    raise RuntimeError("Saved sample index does not match fixed list")
                grid_h = int(saved["grid_h"])
                grid_w = int(saved["grid_w"])
                qnorm_heads_tensor = torch.from_numpy(
                    saved["qnorm_heads"].copy()
                ).float()
                expected_maps = saved["qnorm_maps"].copy()
                if not np.array_equal(
                    qnorm_heads_tensor.numpy().reshape(
                        qnorm_heads_tensor.shape[0], grid_h, grid_w
                    ),
                    expected_maps,
                ):
                    raise RuntimeError("Saved raw [H, N] and [H, grid_h, grid_w] disagree")
            if (student_grid_h, student_grid_w) != (grid_h, grid_w):
                raise RuntimeError(
                    "Teacher saved grid {}x{} and Student grid {}x{} disagree".format(
                        grid_h, grid_w, student_grid_h, student_grid_w
                    )
                )
            images = image.unsqueeze(0).to(device)
            cams = torch.tensor([camid], dtype=torch.long, device=device)
            views = torch.tensor([viewid], dtype=torch.long, device=device)
            initial_student.eval()
            with torch.inference_mode():
                _, student_aux = initial_student(
                    images, cam_label=cams, view_label=views, return_aux=True
                )
            student_cls_attn = student_aux["last_cls_attn"]
            if student_cls_attn.ndim != 3 or student_cls_attn.shape[0] != 1:
                raise RuntimeError("Expected Student last_cls_attn with shape [1, H, N]")
            student_cls_attn_heads = student_cls_attn[0].float().cpu()
            if student_cls_attn_heads.shape[-1] != qnorm_heads_tensor.shape[-1]:
                raise RuntimeError("Teacher and Student patch counts disagree")
            consensus_statistics = compute_consensus_statistics(
                qnorm_heads_tensor,
                student_cls_attn_heads,
                args.t_mean,
                args.t_std,
            )
            rgb = denormalize(image, cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
            render_consensus_sample(
                rgb,
                qnorm_heads_tensor,
                consensus_statistics,
                grid_h,
                grid_w,
                sample_dir,
                args.t_mean,
                args.t_std,
            )
            np.savez_compressed(
                sample_dir / "qnorm_consensus_analysis.npz",
                z_heads=consensus_statistics["z_heads"].numpy(),
                z_mean=consensus_statistics["z_mean"].numpy(),
                z_std=consensus_statistics["z_std"].numpy(),
                low_score=consensus_statistics["low_score"].numpy(),
                agreement=consensus_statistics["agreement"].numpy(),
                consensus=consensus_statistics["consensus"].numpy(),
                student_cls_attn_heads=consensus_statistics[
                    "student_cls_attn_heads"
                ].numpy(),
                student_cls_before=consensus_statistics[
                    "student_cls_before"
                ].numpy(),
                cls_times_consensus=consensus_statistics[
                    "cls_times_consensus"
                ].numpy(),
                consensus_mass=np.asarray(
                    float(consensus_statistics["consensus_mass"]),
                    dtype=np.float32,
                ),
                t_mean=np.asarray(args.t_mean, dtype=np.float32),
                t_std=np.asarray(args.t_std, dtype=np.float32),
                grid_h=np.asarray(grid_h, dtype=np.int64),
                grid_w=np.asarray(grid_w, dtype=np.int64),
                sample_index=np.asarray(sample_index, dtype=np.int64),
            )
            consensus_rows.append({
                "order": global_order,
                "sample_index": sample_index,
                "pid": pid,
                "camid": camid,
                "viewid": viewid,
                "image_path": str(image_path),
                "t_mean": args.t_mean,
                "t_std": args.t_std,
                "consensus_mass": float(
                    consensus_statistics["consensus_mass"]
                ),
                "consensus_mean": float(
                    consensus_statistics["consensus"].mean()
                ),
                "z_mean_abs_mean": float(
                    consensus_statistics["z_mean"].abs().mean()
                ),
                "z_std_mean": float(consensus_statistics["z_std"].mean()),
            })
            print(
                "[{}/{}] analyzed index={} path={} heads={} grid={}x{} "
                "ConsensusMass={:.6f}".format(
                    global_order - args.start_sample, len(selected_indices),
                    sample_index, image_path, qnorm_heads_tensor.shape[0],
                    grid_h, grid_w,
                    float(consensus_statistics["consensus_mass"]),
                ),
                flush=True,
            )
        metrics_name = "consensus_metrics_samples_{:02d}_{:02d}.csv".format(
            args.start_sample + 1, stop_sample
        )
        write_csv(output_dir / metrics_name, consensus_rows)
        print(
            "Consensus analysis complete; raw Teacher artifacts were not rewritten: {}".format(
                output_dir
            ),
            flush=True,
        )
        return

    device = resolve_device(args.device)
    teacher = load_frozen_teacher(
        cfg, num_classes, camera_num, view_num,
        args.teacher_checkpoint, device,
    )
    grid_h = int(teacher.base.patch_embed.num_y)
    grid_w = int(teacher.base.patch_embed.num_x)
    output_dir.mkdir(parents=True, exist_ok=False)
    all_metrics: List[Dict[str, object]] = []
    manifest: List[Dict[str, object]] = []
    cosine_matrices = []
    iou_matrices = []

    for order, sample_index in enumerate(
        selected_indices, start=args.start_sample + 1
    ):
        image, pid, camid, viewid, image_path = dataset[sample_index]
        images = image.unsqueeze(0).to(device)
        cams = torch.tensor([camid], dtype=torch.long, device=device)
        views = torch.tensor([viewid], dtype=torch.long, device=device)
        teacher.eval()
        with torch.inference_mode():
            _, auxiliary = teacher(
                images, cam_label=cams, view_label=views, return_aux=True
            )
        qnorm = auxiliary["last_qnorm"]
        if qnorm.ndim != 3 or qnorm.shape[0] != 1:
            raise RuntimeError("Expected last_qnorm with shape [1, H, N]")
        qnorm_heads_tensor = qnorm[0].float().cpu()
        heads, patches = qnorm_heads_tensor.shape
        if grid_h * grid_w != patches:
            raise RuntimeError(
                "Patch grid {}x{} does not match N={}".format(
                    grid_h, grid_w, patches
                )
            )
        qnorm_maps_tensor = qnorm_heads_tensor.reshape(heads, grid_h, grid_w)
        statistics_tensor = compute_statistics(qnorm_heads_tensor)

        qnorm_heads = qnorm_heads_tensor.numpy()
        qnorm_maps = qnorm_maps_tensor.numpy()
        statistics = {
            key: value.cpu().numpy() for key, value in statistics_tensor.items()
        }
        rgb = denormalize(image, cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
        sample_dir = output_dir / "sample_{:02d}_idx_{:05d}".format(
            order, sample_index
        )
        sample_dir.mkdir(parents=False, exist_ok=False)
        render_sample_figures(
            rgb, qnorm_heads_tensor, grid_h, grid_w,
            statistics, sample_dir, sample_index,
        )
        np.savez_compressed(
            sample_dir / "qnorm_analysis.npz",
            qnorm_heads=qnorm_heads,
            qnorm_maps=qnorm_maps,
            mean_map=statistics["mean_map"].reshape(grid_h, grid_w),
            std_map=statistics["std_map"].reshape(grid_h, grid_w),
            low_mean_low_std_map=statistics["low_mean_low_std_map"].reshape(
                grid_h, grid_w
            ),
            cosine_matrix=statistics["cosine_matrix"],
            top20_iou_matrix=statistics["top20_iou_matrix"],
            sample_index=np.asarray(sample_index, dtype=np.int64),
            pid=np.asarray(pid, dtype=np.int64),
            camid=np.asarray(camid, dtype=np.int64),
            viewid=np.asarray(viewid, dtype=np.int64),
            image_path=np.asarray(str(image_path)),
            grid_h=np.asarray(grid_h, dtype=np.int64),
            grid_w=np.asarray(grid_w, dtype=np.int64),
        )
        rows = write_head_metrics(
            sample_dir / "head_metrics.csv", sample_index, statistics
        )
        for row in rows:
            row.update({
                "pid": pid,
                "camid": camid,
                "viewid": viewid,
                "image_path": str(image_path),
            })
        all_metrics.extend(rows)
        manifest.append({
            "order": order,
            "sample_index": sample_index,
            "pid": pid,
            "camid": camid,
            "viewid": viewid,
            "image_path": str(image_path),
            "output_directory": sample_dir.name,
        })
        cosine_matrices.append(statistics["cosine_matrix"])
        iou_matrices.append(statistics["top20_iou_matrix"])
        print(
            "[{}/{}] index={} path={} heads={} grid={}x{}".format(
                order - args.start_sample, len(selected_indices), sample_index,
                image_path, heads, grid_h, grid_w
            ),
            flush=True,
        )

    write_csv(output_dir / "fixed_samples_manifest.csv", manifest)
    write_csv(output_dir / "head_metrics_all_samples.csv", all_metrics)
    mean_cosine = np.stack(cosine_matrices).mean(axis=0)
    mean_iou = np.stack(iou_matrices).mean(axis=0)
    np.savez_compressed(
        output_dir / "aggregate_head_matrices.npz",
        sample_indices=np.asarray(selected_indices, dtype=np.int64),
        cosine_matrices=np.stack(cosine_matrices),
        top20_iou_matrices=np.stack(iou_matrices),
        mean_cosine_matrix=mean_cosine,
        mean_top20_iou_matrix=mean_iou,
    )
    save_matrix_pair(
        mean_cosine, mean_iou,
        output_dir / "aggregate_head_similarity_matrices.png",
        "Mean across {} fixed validation samples".format(len(selected_indices)),
    )
    metadata = {
        "analysis_only": True,
        "teacher_eval": True,
        "teacher_parameters_frozen": True,
        "config_file": args.config_file,
        "teacher_checkpoint": args.teacher_checkpoint,
        "device": str(device),
        "fixed_sample_indices": list(selected_indices),
        "grid_h": grid_h,
        "grid_w": grid_w,
        "num_heads": int(mean_cosine.shape[0]),
        "top20_k": max(1, int(grid_h * grid_w * 0.20)),
        "spatial_entropy_definition": "-sum((qnorm/sum(qnorm))*log(qnorm/sum(qnorm)))",
        "top20_concentration_definition": "sum(top20% qnorm)/sum(all qnorm)",
        "low_mean_low_std_definition": (
            "(1-minmax(head_mean_map))*(1-minmax(head_std_map))"
        ),
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Analysis complete: {}".format(output_dir), flush=True)


if __name__ == "__main__":
    main()
