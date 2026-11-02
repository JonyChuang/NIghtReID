#!/usr/bin/env python3
"""Detailed layer/head visualizations from the frozen teacher or saved maps.

Run from the TransReID project root. --rerender_from reads existing numeric
artifacts without importing torch or loading a model. Every run writes to a
new output directory; the original script and saved results remain intact.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
PANELS = (
    ("qnorm", "Patch Q norm", False),
    ("attention", "CLS to patch attention", False),
    ("value_norm", "Patch Value norm", False),
    ("value_cos", "CLS/patch Value cosine", True),
    ("contribution", "Patch contribution to CLS", False),
    ("token_cos", "CLS/patch cosine after block", True),
)
HEAD_PANELS = {key: (title, signed) for key, title, signed in PANELS
               if key != "token_cos"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_file", default="experiment_variants/qnorm_context_rank/config.yml")
    parser.add_argument("--teacher_checkpoint", default="../../log/transreid/night600/vit_base_ics_teacher/transformer_teacher_120.pth")
    parser.add_argument("--output_dir", help="A new directory; existing directories are never overwritten")
    parser.add_argument("--sample_indices", help="Dataset indices; default 245,4472,6022 for inference, all cached samples for rerender")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rerender_from", help="Existing result directory containing manifest.json and sample_*/raw_layer_maps.npz")
    parser.add_argument("--image_size", help="Height,width for legacy caches without saved RGB; otherwise use the saved config")
    parser.add_argument("--per_head", dest="per_head", action="store_true",
                        help="Render head figures (enabled by default)")
    parser.add_argument("--no_per_head", dest="per_head", action="store_false")
    parser.set_defaults(per_head=True)
    parser.add_argument("--head_layers", default="all", help="all or one-based blocks, e.g. 7,8,9,10,11,12")
    parser.add_argument("--head_metrics", default="qnorm,attention,value_cos,contribution",
                        help="Comma-separated metrics; also accepts value_norm")
    parser.add_argument("--display_gamma", type=float, default=1.0,
                        help="Display only: a value below 1 brightens the image; model inputs and maps are unchanged")
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()
    if not np.isfinite(args.display_gamma) or args.display_gamma <= 0:
        parser.error("display_gamma must be finite and positive")
    if args.dpi <= 0:
        parser.error("dpi must be positive")
    args.head_metrics = [item.strip() for item in args.head_metrics.split(",") if item.strip()]
    if not args.head_metrics or any(key not in HEAD_PANELS for key in args.head_metrics):
        parser.error("head_metrics must contain qnorm, attention, value_norm, value_cos or contribution")
    args.head_metrics = list(dict.fromkeys(args.head_metrics))
    return args


def integer_list(text, label, unique=True):
    try:
        values = [int(item.strip()) for item in text.split(",")]
    except ValueError as error:
        raise ValueError("%s must be comma-separated integers" % label) from error
    if not values or (unique and len(values) != len(set(values))):
        raise ValueError("%s must be nonempty%s" % (label, " and contain no duplicates" if unique else ""))
    return values


def validate_arrays(arrays):
    grid = tuple(int(item) for item in arrays["grid"])
    if len(grid) != 2 or min(grid) <= 0:
        raise ValueError("Expected a positive two-dimensional patch grid")
    shape = arrays["qnorm"].shape
    if len(shape) != 2 or shape[0] < 1 or shape[1] != grid[0] * grid[1]:
        raise ValueError("Layer maps do not match the saved patch grid")
    for key, _, _ in PANELS:
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError("Invalid shape or nonfinite values in " + key)
    for key in HEAD_PANELS:
        head_values = arrays[key + "_heads"]
        if (head_values.ndim != 3 or head_values.shape[0] != shape[0]
                or head_values.shape[2] != shape[1] or head_values.shape[1] < 1
                or not np.isfinite(head_values).all()):
            raise ValueError("Invalid per-head maps for " + key)
    if np.any(arrays["attention"] < 0) or np.any(arrays["contribution"] < 0):
        raise ValueError("Attention probabilities and contribution norms must be nonnegative")
    return grid


def minmax_map(values):
    values = np.asarray(values, dtype=np.float64)
    low, high = float(values.min()), float(values.max())
    if high - low <= 1e-10:
        return np.zeros_like(values), low, high
    return (values - low) / (high - low), low, high


def observed_limits(values):
    low, high = float(np.min(values)), float(np.max(values))
    if high - low <= 1e-10:
        return low - 1e-6, high + 1e-6
    return low, high


def shared_limits(values, signed):
    if signed:
        return -1.0, 1.0
    return 0.0, max(float(np.percentile(values, 99)), 1e-8)


def mass_distribution(values):
    values = np.asarray(values, dtype=np.float64)
    mass = values.sum(axis=-1, keepdims=True)
    return np.divide(values, mass, out=np.zeros_like(values), where=mass > 0)


def photo(axis, rgb):
    axis.imshow(rgb)
    axis.set_xticks([])
    axis.set_yticks([])


def heatmap(axis, values, grid, low, high, signed=False, rgb=None):
    if rgb is None:
        heat = axis.imshow(values.reshape(grid), cmap="coolwarm" if signed else "turbo",
                           vmin=low, vmax=high, interpolation="nearest")
    else:
        photo(axis, rgb)
        heat = axis.imshow(values.reshape(grid), cmap="coolwarm" if signed else "turbo",
                           vmin=low, vmax=high, interpolation="nearest", alpha=0.65,
                           extent=(-0.5, rgb.shape[1] - 0.5, rgb.shape[0] - 0.5, -0.5))
    axis.set_xticks([])
    axis.set_yticks([])
    return heat


def new_figure(rows, columns, width=18):
    return plt.subplots(rows, columns, figsize=(width, max(4, rows * 2.15)),
                        squeeze=False, constrained_layout=True)


def colorbar(fig, heat, axes, label, extend="neither"):
    bar = fig.colorbar(heat, ax=list(axes), orientation="horizontal", fraction=0.025,
                      pad=0.02, aspect=18, extend=extend)
    bar.set_label(label, fontsize=8)
    bar.ax.tick_params(labelsize=7)


def save_figure(fig, path, dpi):
    try:
        fig.savefig(str(path), dpi=dpi)
    finally:
        plt.close(fig)


def render_overview(arrays, rgb, output_dir, args, relative=False, overlay=False):
    layers = arrays["qnorm"].shape[0]
    grid = tuple(int(item) for item in arrays["grid"])
    display_rgb = np.power(rgb, args.display_gamma)
    fig, axes = new_figure(layers, len(PANELS) + 1)
    for layer in range(layers):
        photo(axes[layer, 0], display_rgb)
        axes[layer, 0].set_ylabel("Block %02d" % (layer + 1), fontsize=9)
    axes[0, 0].set_title("Night image\ndisplay gamma %.2g" % args.display_gamma, fontsize=9)
    for column, (key, title, signed) in enumerate(PANELS, 1):
        low, high = (0.0, 1.0) if relative else shared_limits(arrays[key], signed)
        for layer in range(layers):
            values = arrays[key][layer]
            if relative:
                values, raw_low, raw_high = minmax_map(values)
                axes[layer, column].set_xlabel("raw [%.3g, %.3g]" % (raw_low, raw_high), fontsize=7)
            heat = heatmap(axes[layer, column], values, grid, low, high,
                           signed=signed and not relative, rgb=display_rgb if overlay else None)
        axes[0, column].set_title(title, fontsize=9)
        colorbar(fig, heat, axes[:, column], "Within-layer relative value" if relative else "Raw value",
                 extend="max" if not relative and not signed else "neither")
    if relative:
        fig.suptitle("Each map scaled separately to [0,1]; captions show raw min/max.\n"
                     "Colors compare spatial positions within a map, not magnitude between layers.", fontsize=11)
        filename = "all_layers_overlay_relative.png" if overlay else "all_layers_relative_scale.png"
    else:
        fig.suptitle("One raw scale per column across layers; cosine scale [-1,1].\n"
                     "Heatmaps shown without image blending; attention updates are not identity attribution.", fontsize=11)
        filename = "all_layers_shared_scale.png"
    save_figure(fig, output_dir / filename, args.dpi)


def render_cosine_detail(arrays, rgb, output_dir, args):
    layers = arrays["qnorm"].shape[0]
    grid = tuple(int(item) for item in arrays["grid"])
    fig, axes = new_figure(layers, 5, width=14)
    for layer in range(layers):
        photo(axes[layer, 0], np.power(rgb, args.display_gamma))
        axes[layer, 0].set_ylabel("Block %02d" % (layer + 1), fontsize=9)
    axes[0, 0].set_title("Night image", fontsize=9)
    for offset, key in enumerate(("value_cos", "token_cos")):
        values = arrays[key]
        centered = values - values.mean(axis=1, keepdims=True)
        low, high = observed_limits(values)
        bound = max(float(np.max(np.abs(centered))), 1e-8)
        raw_col, centered_col = 1 + offset * 2, 2 + offset * 2
        for layer in range(layers):
            raw_heat = heatmap(axes[layer, raw_col], values[layer], grid, low, high, signed=True)
            detail_heat = heatmap(axes[layer, centered_col], centered[layer], grid, -bound, bound, signed=True)
            axes[layer, raw_col].set_xlabel("mean %.3g; range [%.3g, %.3g]" %
                                           (values[layer].mean(), values[layer].min(), values[layer].max()), fontsize=7)
        label = "Value cosine" if key == "value_cos" else "Post-block token cosine"
        axes[0, raw_col].set_title(label + "\nraw, observed range", fontsize=9)
        axes[0, centered_col].set_title(label + "\nminus layer spatial mean", fontsize=9)
        colorbar(fig, raw_heat, axes[:, raw_col], "Raw cosine (zoomed scale)")
        colorbar(fig, detail_heat, axes[:, centered_col], "Deviation: red above / blue below layer mean")
    fig.suptitle("Raw cosine uses a shared observed range; centered cosine shows spatial differences.\n"
                 "A small contrast does not establish absence of identity information.", fontsize=11)
    save_figure(fig, output_dir / "all_layers_cosine_detail.png", args.dpi)


def render_layer_difference(arrays, rgb, output_dir, args):
    layers = arrays["qnorm"].shape[0]
    grid = tuple(int(item) for item in arrays["grid"])
    fig, axes = new_figure(layers, 5, width=14)
    for layer in range(layers):
        photo(axes[layer, 0], np.power(rgb, args.display_gamma))
        axes[layer, 0].set_ylabel("Block %02d" % (layer + 1), fontsize=9)
    axes[0, 0].set_title("Night image", fontsize=9)
    for offset, key in enumerate(("attention", "contribution")):
        distribution = mass_distribution(arrays[key])
        difference = distribution - distribution[-1:]
        bound = max(float(np.max(np.abs(difference))), 1e-8)
        high = max(float(distribution.max()), 1e-8)
        distribution_col, difference_col = 1 + offset * 2, 2 + offset * 2
        for layer in range(layers):
            raw_heat = heatmap(axes[layer, distribution_col], distribution[layer], grid, 0, high)
            diff_heat = heatmap(axes[layer, difference_col], difference[layer], grid, -bound, bound, signed=True)
        title = "Attention: patch mass" if key == "attention" else "Contribution: norm mass"
        axes[0, distribution_col].set_title(title + "\nsum over patches = 1", fontsize=9)
        axes[0, difference_col].set_title("This block minus block %02d\nred: more / blue: less" % layers, fontsize=9)
        colorbar(fig, raw_heat, axes[:, distribution_col], "Fraction of patch attention / contribution-norm mass")
        colorbar(fig, diff_heat, axes[:, difference_col], "Spatial mass difference from final block")
    fig.suptitle("Each layer's patch values divided by their sum; compare spatial allocation with the final block.\n"
                 "Norm fractions are not signed or additive identity contributions.", fontsize=11)
    save_figure(fig, output_dir / "layer_vs_last.png", args.dpi)


def render_heads(arrays, rgb, output_dir, args):
    layers = arrays["qnorm"].shape[0]
    indices = list(range(layers)) if args.head_layers == "all" else [
        item - 1 for item in integer_list(args.head_layers, "head_layers")]
    if any(index < 0 or index >= layers for index in indices):
        raise ValueError("head_layers must lie between 1 and %d" % layers)
    grid = tuple(int(item) for item in arrays["grid"])
    for key in args.head_metrics:
        title, signed = HEAD_PANELS[key]
        values = arrays[key + "_heads"]
        heads = values.shape[1]
        for relative in (False, True):
            low, high = (0.0, 1.0) if relative else shared_limits(values[indices], signed)
            fig, axes = new_figure(len(indices), heads + 1, width=max(14, (heads + 1) * 1.4))
            for row, layer in enumerate(indices):
                photo(axes[row, 0], np.power(rgb, args.display_gamma))
                axes[row, 0].set_ylabel("Block %02d" % (layer + 1), fontsize=9)
                for head in range(heads):
                    head_map = values[layer, head]
                    if relative:
                        head_map, raw_low, raw_high = minmax_map(head_map)
                        axes[row, head + 1].set_xlabel("[%.3g, %.3g]" % (raw_low, raw_high), fontsize=6)
                    heat = heatmap(axes[row, head + 1], head_map, grid, low, high, signed=signed and not relative)
                    if row == 0:
                        axes[row, head + 1].set_title("Head %02d" % (head + 1), fontsize=8)
            axes[0, 0].set_title("Night image", fontsize=8)
            if relative:
                fig.suptitle(title + " | each head/layer scaled independently; labels show raw min/max.\n"
                             "Within-map spatial contrast only; colors do not compare head magnitudes.", fontsize=11)
            else:
                fig.suptitle(title + " | one raw scale across the displayed heads and layers.", fontsize=11)
            colorbar(fig, heat, axes[:, 1:].flat, "Relative value" if relative else "Raw value",
                     extend="max" if not relative and not signed else "neither")
            filename = "per_head_%s_%s.png" % (key, "relative" if relative else "shared")
            save_figure(fig, output_dir / filename, args.dpi)


def layer_metrics(arrays, sample_index):
    rows = []
    for layer in range(arrays["qnorm"].shape[0]):
        row = {"sample_index": sample_index, "layer": layer + 1,
               "cls_to_patch_attention_mass": float(arrays["attention"][layer].sum()),
               "cls_self_attention_mass": float(arrays["cls_self_attention_heads"][layer].mean()),
               "patch_contribution_norm_sum": float(arrays["contribution"][layer].sum()),
               "reconstruction_error": float(arrays["reconstruction_error"][layer])}
        for key, _, _ in PANELS:
            values = arrays[key][layer]
            for name, value in (("min", values.min()), ("mean", values.mean()),
                                ("max", values.max()), ("std", values.std())):
                row[key + "_" + name] = float(value)
        rows.append(row)
    return rows


def save_sample(arrays, rgb, sample, output_dir, args):
    validate_arrays(arrays)
    sample_dir = output_dir / ("sample_%05d" % sample["index"])
    sample_dir.mkdir()
    np.savez_compressed(str(sample_dir / "raw_layer_maps.npz"),
                        **{key: value for key, value in arrays.items() if key != "rgb"}, rgb=rgb)
    plt.imsave(str(sample_dir / "input_night.png"), rgb)
    render_overview(arrays, rgb, sample_dir, args)
    render_overview(arrays, rgb, sample_dir, args, relative=True)
    render_overview(arrays, rgb, sample_dir, args, relative=True, overlay=True)
    render_cosine_detail(arrays, rgb, sample_dir, args)
    render_layer_difference(arrays, rgb, sample_dir, args)
    if args.per_head:
        render_heads(arrays, rgb, sample_dir, args)
    print("Saved visualization for index %d -> %s" % (sample["index"], sample_dir), flush=True)
    return layer_metrics(arrays, sample["index"])


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def cached_image_size(manifest, args):
    if args.image_size:
        size = integer_list(args.image_size, "image_size", unique=False)
    else:
        size = manifest.get("input_size")
        if size is None:
            import yaml
            config_path = project_path(manifest.get("config", args.config_file))
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            size = config.get("INPUT", {}).get("SIZE_TEST")
    if size is None or len(size) != 2 or min(size) <= 0:
        raise ValueError("Cannot recover legacy input size; pass --image_size height,width")
    return int(size[0]), int(size[1])


def rerender(args, output_dir):
    source_dir = Path(args.rerender_from).resolve()
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    wanted = None if args.sample_indices is None else integer_list(args.sample_indices, "sample_indices")
    samples = manifest["samples"]
    if wanted is not None:
        by_index = {sample["index"]: sample for sample in samples}
        missing = set(wanted).difference(by_index)
        if missing:
            raise ValueError("Requested samples not in cache: %s" % sorted(missing))
        samples = [by_index[index] for index in wanted]
    if not samples:
        raise ValueError("No cached samples found")
    rows = []
    image_size = None
    for sample in samples:
        source = source_dir / ("sample_%05d" % sample["index"]) / "raw_layer_maps.npz"
        with np.load(str(source), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if "rgb" in arrays:
            rgb = arrays.pop("rgb")
        else:
            if image_size is None:
                image_size = cached_image_size(manifest, args)
            height, width = image_size
            image_path = project_path(sample["image_path"])
            # Matches the legacy validation transform: PIL bilinear resize.
            with Image.open(str(image_path)) as image:
                resized = image.convert("RGB").resize((width, height), Image.BILINEAR)
                rgb = np.asarray(resized, dtype=np.float32) / 255.0
        rows.extend(save_sample(arrays, rgb, sample, output_dir, args))
    manifest.update({"samples": samples, "indices": [sample["index"] for sample in samples],
                     "render_source": str(source_dir), "mode": "rerender_saved_maps",
                     "input_size": list(rgb.shape[:2])})
    return manifest, rows


def frozen_inference(args, output_dir):
    # Keep model imports inside this branch: cached rerender never imports torch
    # or calls the teacher. Reuse the verified QKV/contribution capture unchanged.
    import torch
    from config import cfg as base_cfg
    from datasets import make_dataloader
    from analyze_teacher_qnorm_heads import load_frozen_teacher, resolve_device
    from visualize_cls_value_layers import Capture, pack

    indices = integer_list(args.sample_indices or "245,4472,6022", "sample_indices")
    if min(indices) < 0:
        raise ValueError("sample_indices must be nonnegative")
    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.JPM = False
    cfg.MODEL.PRETRAIN_CHOICE = ""
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    _, _, val_loader, _, num_classes, camera_num, view_num = make_dataloader(cfg)
    dataset = val_loader.dataset
    if max(indices) >= len(dataset):
        raise IndexError("Requested index exceeds query+gallery dataset length")
    device = resolve_device(args.device)
    model = load_frozen_teacher(cfg, num_classes, camera_num, view_num,
                                args.teacher_checkpoint, device)
    grid = (int(model.base.patch_embed.num_y), int(model.base.patch_embed.num_x))
    manifest = {"checkpoint": args.teacher_checkpoint, "config": args.config_file,
                "indices": indices, "device": str(device), "grid": grid,
                "input_size": list(cfg.INPUT.SIZE_TEST), "mode": "frozen_teacher_inference", "samples": []}
    rows = []
    capture = Capture(model.base)
    try:
        for index in indices:
            image, pid, camid, viewid, image_path = dataset[index]
            capture.clear()
            with torch.inference_mode():
                model(image.unsqueeze(0).to(device), cam_label=torch.tensor([camid], device=device),
                      view_label=torch.tensor([viewid], device=device))
            arrays = pack(capture, grid)
            mean = torch.tensor(cfg.INPUT.PIXEL_MEAN).view(3, 1, 1)
            std = torch.tensor(cfg.INPUT.PIXEL_STD).view(3, 1, 1)
            rgb = (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            sample = {"index": index, "pid": int(pid), "camid": int(camid),
                      "viewid": int(viewid), "image_path": str(image_path)}
            rows.extend(save_sample(arrays, rgb, sample, output_dir, args))
            manifest["samples"].append(sample)
    finally:
        capture.close()
    return manifest, rows


def main():
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.rerender_from:
        source = Path(args.rerender_from).resolve()
        output_dir = source.parent / (source.name + "_v2_" + stamp)
    else:
        output_dir = Path("../../log/transreid/night600/cls_value_layers_v2_" + stamp)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest, rows = rerender(args, output_dir) if args.rerender_from else frozen_inference(args, output_dir)
    manifest["visualization"] = {"script": Path(__file__).name, "display_gamma": args.display_gamma,
                                 "per_head": args.per_head, "head_layers": args.head_layers,
                                 "head_metrics": args.head_metrics,
                                 "shared_positive_scale": "[0, 99th percentile across displayed layers/heads]",
                                 "shared_cosine_scale": [-1, 1],
                                 "relative_scale": "(value - spatial min) / (spatial max - spatial min), per map",
                                 "difference": "layer patch mass distribution minus final-layer distribution"}
    manifest["note"] = "These maps show layer/head spatial responses; they do not establish identity-information gain."
    with (output_dir / "layer_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Complete:", output_dir.resolve())


if __name__ == "__main__":
    main()
