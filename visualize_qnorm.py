#!/usr/bin/env python3
"""Visualize Q/K/V vector norms from a TransReID attention block.

This script intentionally uses a forward-pre-hook, so the TransReID source does
not need to be changed. It expects to be run from the project root, where the
repository's ``config``, ``model`` and other packages are importable.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="TransReID YAML config")
    parser.add_argument("--checkpoint", required=True, help="trained .pth/.pth.tar")
    parser.add_argument("--images", required=True, help="image file, directory, or text list")
    parser.add_argument("--output", default="qnorm_vis", help="output directory")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="only needed when it cannot be inferred from the checkpoint")
    parser.add_argument("--camera-num", type=int, default=None,
                        help="camera count; defaults to 8 when cfg.MODEL.SIE_CAMERA is enabled")
    parser.add_argument("--view-num", type=int, default=None,
                        help="view count passed to TransReID make_model")
    parser.add_argument("--camera-label", type=int, default=0,
                        help="camera label used for SIE during visualization")
    parser.add_argument("--view-label", type=int, default=0,
                        help="view label used for SIE during visualization")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Transformer block index; -1 means the last block")
    parser.add_argument("--input-height", type=int, default=256)
    parser.add_argument("--input-width", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--prefix-tokens", type=int, default=1,
                        help="CLS/distillation tokens before patch tokens")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha", type=float, default=0.48,
                        help="heatmap overlay opacity")
    parser.add_argument("--percentile", type=float, default=99.0,
                        help="upper percentile used for robust heatmap scaling")
    parser.add_argument("--save-npy", action="store_true",
                        help="also save raw [16, 8] Q/K/V/attention maps as NumPy files")
    parser.add_argument("--model-module", default="model",
                        help="module containing make_model(cfg, num_classes)")
    parser.add_argument("--attention-name", default=None,
                        help="exact named_modules path; useful when MODEL.JPM is enabled")
    return parser.parse_args()


def load_cfg(config_path: str):
    """Load the repository config without assuming a particular yacs import path."""
    config_mod = importlib.import_module("config")
    cfg = getattr(config_mod, "cfg", None)
    if cfg is None:
        raise RuntimeError("config.py must expose cfg")
    if hasattr(cfg, "merge_from_file"):
        cfg.merge_from_file(config_path)
    else:
        raise RuntimeError("The project's cfg does not implement merge_from_file")
    if hasattr(cfg, "freeze"):
        cfg.freeze()
    return cfg


def checkpoint_state(path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint must contain a state-dict or a dict under model/state_dict")
    state = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        clean_key = key[7:] if key.startswith("module.") else key
        state[clean_key] = value
    return state


def infer_num_classes(state: Dict[str, torch.Tensor]) -> Optional[int]:
    candidates = []
    for key, value in state.items():
        if value.ndim == 2 and value.shape[0] > 1 and (
            key.endswith("classifier.weight") or key.endswith("classifier.classifier.weight")
        ):
            candidates.append(int(value.shape[0]))
    return candidates[0] if candidates else None


def make_model(cfg, num_classes: int, module_name: str, camera_num: int, view_num: int):
    mod = importlib.import_module(module_name)
    factory = getattr(mod, "make_model", None)
    if factory is None:
        raise RuntimeError(f"{module_name} does not expose make_model")
    # TransReID-SSL uses make_model(cfg, num_class, camera_num, view_num),
    # while a few forks expose the shorter two-argument factory.
    try:
        return factory(cfg, num_classes, camera_num, view_num)
    except TypeError as four_arg_error:
        try:
            return factory(cfg, num_classes)
        except TypeError:
            raise four_arg_error


def find_attention(model: torch.nn.Module, layer: int, explicit_name: Optional[str] = None) -> Tuple[str, torch.nn.Module]:
    modules = list(model.named_modules())
    if explicit_name:
        for name, mod in modules:
            if name == explicit_name:
                if not hasattr(mod, "qkv"):
                    raise RuntimeError(f"{explicit_name} exists but has no qkv")
                return name, mod
        raise RuntimeError(f"Attention module not found: {explicit_name}")
    matches = [(name, mod) for name, mod in modules if name.endswith(".attn") and hasattr(mod, "qkv")]
    if not matches:
        raise RuntimeError("Could not find an attention module exposing qkv")
    if layer < 0:
        # With JPM, b1 is the global final block. b2 sees rearranged local
        # tokens and cannot be reshaped into the original 16 x 8 image grid.
        global_jpm = [(name, mod) for name, mod in matches if ".b1." in name]
        if global_jpm:
            return global_jpm[-1]
        return matches[-1]
    exact = [(name, mod) for name, mod in matches if f"blocks.{layer}.attn" in name]
    if exact:
        return exact[-1]
    raise RuntimeError("No attention module matched block index %d; found: %s" %
                       (layer, ", ".join(name for name, _ in matches)))


def image_paths(spec: str) -> List[Path]:
    path = Path(spec)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if path.is_file():
        paths = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                candidate = Path(line.split()[0])
                if candidate.is_file():
                    paths.append(candidate)
        return paths
    raise FileNotFoundError(spec)


def make_transform(height: int, width: int, mean=None, std=None):
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def split_qkv(module: torch.nn.Module, x: torch.Tensor):
    b, n, c = x.shape
    qkv = module.qkv(x)
    heads = int(module.num_heads)
    if qkv.shape[-1] != 3 * c:
        raise RuntimeError(f"Unexpected qkv shape {tuple(qkv.shape)} for input {tuple(x.shape)}")
    head_dim = qkv.shape[-1] // (3 * heads)
    qkv = qkv.reshape(b, n, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
    return qkv[0], qkv[1], qkv[2]


def norm_map(x: torch.Tensor, prefix_tokens: int, grid: Tuple[int, int]) -> torch.Tensor:
    values = x.norm(dim=-1).mean(dim=1)
    values = values[:, prefix_tokens:]
    expected = grid[0] * grid[1]
    if values.shape[-1] != expected:
        raise RuntimeError("Token count %d does not match grid %s after removing %d prefix token(s). "
                           "Pass --prefix-tokens and --input-height/width explicitly." %
                           (values.shape[-1], grid, prefix_tokens))
    return values.reshape(values.shape[0], *grid)


def attention_map(q: torch.Tensor, k: torch.Tensor, module: torch.nn.Module,
                  prefix_tokens: int, grid: Tuple[int, int]) -> torch.Tensor:
    logits = (q @ k.transpose(-2, -1)) * float(getattr(module, "scale", 1.0))
    attn = logits.softmax(dim=-1)
    if prefix_tokens:
        values = attn[:, :, :prefix_tokens, prefix_tokens:].mean(dim=(1, 2))
    else:
        values = attn.mean(dim=(1, 2))
    expected = grid[0] * grid[1]
    if values.shape[-1] != expected:
        raise RuntimeError("Attention token count does not match grid %s" % (grid,))
    return values.reshape(values.shape[0], *grid)


def robust_normalize(values: torch.Tensor, percentile: float) -> torch.Tensor:
    flat = values.flatten()
    lo = flat.min()
    hi = torch.quantile(flat, percentile / 100.0)
    return ((values - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)


def save_panel(original: Image.Image, maps: Dict[str, torch.Tensor], output: Path,
               alpha: float, percentile: float):
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(maps) + 1, figsize=(4 * (len(maps) + 1), 4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(original)
    axes[0].set_title("Input")
    axes[0].axis("off")
    base = np.asarray(original.convert("RGB"), dtype=np.float32) / 255.0
    for axis, (name, value) in zip(axes[1:], maps.items()):
        norm = robust_normalize(value, percentile).cpu()[None, None]
        resized = F.interpolate(norm, size=(original.height, original.width), mode="bilinear",
                                align_corners=False)[0, 0].numpy()
        axis.imshow(base)
        axis.imshow(resized, cmap="turbo", vmin=0, vmax=1, alpha=alpha)
        axis.set_title(name)
        axis.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    state = checkpoint_state(args.checkpoint)
    num_classes = args.num_classes or infer_num_classes(state)
    if num_classes is None:
        raise RuntimeError("Cannot infer num_classes; pass --num-classes")
    cfg = load_cfg(args.config)
    model_cfg = getattr(cfg, "MODEL", None)
    sie_camera = bool(getattr(model_cfg, "SIE_CAMERA", False))
    sie_view = bool(getattr(model_cfg, "SIE_VIEW", False))
    camera_num = args.camera_num if args.camera_num is not None else (8 if sie_camera else 0)
    view_num = args.view_num if args.view_num is not None else (1 if sie_view else 0)
    model = make_model(cfg, num_classes, args.model_module, camera_num, view_num)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("Warning: missing checkpoint keys:", len(missing))
    if unexpected:
        print("Warning: unexpected checkpoint keys:", len(unexpected))
    model.to(device).eval()
    attention_name, attention = find_attention(model, args.layer, args.attention_name)
    print(f"Hooking {attention_name}; num_classes={num_classes}")

    captured: Dict[str, torch.Tensor] = {}
    def pre_hook(module, inputs):
        x = inputs[0]
        with torch.no_grad():
            q, k, v = split_qkv(module, x)
            captured["q"] = q.float().detach()
            captured["k"] = k.float().detach()
            captured["v"] = v.float().detach()
            captured["attn"] = attention_map(q, k, module, args.prefix_tokens, grid)
    handle = attention.register_forward_pre_hook(pre_hook)

    grid = (args.input_height // args.patch_size, args.input_width // args.patch_size)
    input_cfg = getattr(cfg, "INPUT", None)
    mean = getattr(input_cfg, "PIXEL_MEAN", [0.485, 0.456, 0.406])
    std = getattr(input_cfg, "PIXEL_STD", [0.229, 0.224, 0.225])
    transform = make_transform(args.input_height, args.input_width, mean, std)
    paths = image_paths(args.images)
    if args.max_images is not None:
        paths = paths[:args.max_images]
    if not paths:
        raise RuntimeError("No images found")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for index, path in enumerate(paths):
            original = Image.open(path).convert("RGB")
            batch = transform(original).unsqueeze(0).to(device)
            captured.clear()
            with torch.no_grad():
                try:
                    model(batch, cam_label=torch.tensor([args.camera_label], device=device),
                          view_label=torch.tensor([args.view_label], device=device))
                except TypeError:
                    # Older forks do not expose camera/view keyword arguments.
                    model(batch)
            maps = {
                "Q-Norm": norm_map(captured["q"], args.prefix_tokens, grid)[0],
                "K-Norm": norm_map(captured["k"], args.prefix_tokens, grid)[0],
                "V-Norm": norm_map(captured["v"], args.prefix_tokens, grid)[0],
                "Attention": captured["attn"][0],
            }
            if args.save_npy:
                raw = {name.lower().replace("-", "_"): value.cpu().numpy()
                       for name, value in maps.items()}
                np.savez(output_dir / f"{index:05d}_{path.stem}.npz", **raw)
            save_panel(original, maps, output_dir / f"{index:05d}_{path.stem}.png",
                       args.alpha, args.percentile)
            print(f"[{index + 1}/{len(paths)}] {path}")
    finally:
        handle.remove()

def _normalize_heatmap(array):
    array = np.asarray(array, dtype=np.float32)
    low, high = np.percentile(array, [1.0, 99.0])
    if high <= low:
        return np.zeros_like(array)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def _resize_heatmap(array, height, width):
    tensor = torch.as_tensor(array, dtype=torch.float32)[None, None]
    resized = F.interpolate(
        tensor,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0].numpy()


def _draw_overlay(axis, image, heatmap, title):
    axis.imshow(image)
    axis.imshow(heatmap, cmap="turbo", alpha=0.58, vmin=0.0, vmax=1.0)
    axis.set_title(title, fontsize=10)
    axis.axis("off")


def save_context_visualizations(
    images,
    maps,
    output_dir,
    epoch,
    grid_size=(16, 8),
    max_images=6,
    pixel_mean=(0.485, 0.456, 0.406),
    pixel_std=(0.229, 0.224, 0.225),
):
    output_dir = Path(output_dir) / f"epoch_{epoch:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_height, grid_width = grid_size
    num_patches = maps["teacher_qnorm"].shape[1]
    if grid_height * grid_width != num_patches:
        raise RuntimeError(
            f"grid_size={grid_size} contains {grid_height * grid_width} cells, "
            f"but the model produced {num_patches} patches."
        )

    mean = torch.tensor(pixel_mean).view(3, 1, 1)
    std = torch.tensor(pixel_std).view(3, 1, 1)
    map_items = [
        ("teacher_qnorm", "Teacher Q-Norm"),
        ("student_patch_norm", "Student Patch Norm"),
        ("pooling_weight", "Q-Norm Pooling Weight"),
        ("context_attention", "CLS-to-Context Attention"),
        ("combined_contribution", "Combined Contribution"),
    ]

    count = min(images.shape[0], max_images)
    for index in range(count):
        image = images[index].detach().cpu().float()
        image = (image * std + mean).clamp(0.0, 1.0)
        image = image.permute(1, 2, 0).numpy()
        image_height, image_width = image.shape[:2]

        figure, axes = plt.subplots(1, 6, figsize=(18, 5))
        axes[0].imshow(image)
        axes[0].set_title("Original", fontsize=10)
        axes[0].axis("off")

        for axis, (name, title) in zip(axes[1:], map_items):
            heatmap = maps[name][index].reshape(grid_height, grid_width).numpy()
            heatmap = _normalize_heatmap(heatmap)
            heatmap = _resize_heatmap(heatmap, image_height, image_width)
            _draw_overlay(axis, image, heatmap, title)

        figure.tight_layout()
        save_path = output_dir / f"context_{index:02d}.jpg"
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(figure)

    print(f"[QNormContextVis] saved {count} images to {output_dir}")

if __name__ == "__main__":
    main()
