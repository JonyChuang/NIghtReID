#!/usr/bin/env python3
"""Visualize CLS-to-patch information across the global TransReID blocks.

Run from the TransReID project root. This file only performs frozen inference;
it does not change the model, checkpoint, data, or existing scripts.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from analyze_teacher_qnorm_heads import load_frozen_teacher, resolve_device


PANELS = (
    ("qnorm", "Patch Q norm", "positive"),
    ("attention", "CLS to patch attention", "positive"),
    ("value_norm", "Patch Value norm", "positive"),
    ("value_cos", "CLS/patch Value cosine", "signed"),
    ("contribution", "Patch contribution to CLS", "positive"),
    ("token_cos", "CLS/patch cosine after block", "signed"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_file", default="experiment_variants/qnorm_context_rank/config.yml")
    parser.add_argument("--teacher_checkpoint", default="../../log/transreid/night600/vit_base_ics_teacher/transformer_teacher_120.pth")
    parser.add_argument("--output_dir", default="../../log/transreid/night600/cls_value_layers_20261004")
    parser.add_argument("--sample_indices", default="245,4472,6022",
                        help="Indices in the Night600 query+gallery dataset")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--per_head", action="store_true",
                        help="Render one additional layer-by-head contribution figure per sample")
    return parser.parse_args()


class Capture:
    """Record actual QKV, attention probabilities, attention output, and block output."""

    def __init__(self, backbone):
        self.blocks = list(backbone.blocks)
        self.values = {}
        self.handles = []
        for index, block in enumerate(self.blocks):
            self.handles.append(block.attn.qkv.register_forward_hook(self.qkv_hook(index)))
            self.handles.append(block.attn.attn_drop.register_forward_pre_hook(self.attn_hook(index)))
            self.handles.append(block.attn.register_forward_hook(self.output_hook(index)))
            self.handles.append(block.register_forward_hook(self.block_hook(index)))

    def clear(self):
        self.values.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()

    def qkv_hook(self, index):
        def hook(module, inputs, output):
            self.values[index] = {"qkv": output.detach()}
        return hook

    def attn_hook(self, index):
        def hook(module, inputs):
            self.values[index]["cls_attention"] = inputs[0][0, :, 0].detach()
        return hook

    def output_hook(self, index):
        def hook(module, inputs, output):
            output = output[0] if isinstance(output, tuple) else output
            record = self.values[index]
            qkv = record.pop("qkv")
            batch, tokens, three_dim = qkv.shape
            heads = module.num_heads
            dim = three_dim // 3
            if batch != 1 or three_dim != 3 * dim or dim % heads:
                raise RuntimeError("Unexpected QKV shape at block %d" % (index + 1))
            q, k, v = qkv.float().reshape(batch, tokens, 3, heads, dim // heads).permute(2, 0, 3, 1, 4)
            q = q[0]
            v = v[0]
            a = record.pop("cls_attention").float()
            if a.shape != (heads, tokens):
                raise RuntimeError("Unexpected CLS attention shape at block %d" % (index + 1))

            # For patch i: sum_h A[CLS,i,h] * V[i,h] * W_O[h]. Projection
            # bias is added once to reconstruct the entire CLS attention update.
            weighted_v = a[..., None] * v
            weight = module.proj.weight.float()
            head_weight = weight.reshape(dim, heads, dim // heads)
            head_terms = torch.einsum("hnd,ohd->hno", weighted_v, head_weight)
            terms = head_terms.sum(dim=0)
            reconstructed = terms.sum(dim=0)
            if module.proj.bias is not None:
                reconstructed = reconstructed + module.proj.bias.float()
            observed = output[0, 0].float()
            error = float(((reconstructed - observed).norm() / observed.norm().clamp_min(1e-8)).item())
            if not torch.allclose(reconstructed, observed, rtol=2e-4, atol=2e-5):
                raise RuntimeError("Block %d CLS contribution reconstruction failed (%.5g)" % (index + 1, error))

            record.update({
                "qnorm_heads": q[:, 1:].norm(dim=-1).cpu(),
                "attention_heads": a[:, 1:].cpu(),
                "cls_self_attention_heads": a[:, 0].cpu(),
                "value_norm_heads": v[:, 1:].norm(dim=-1).cpu(),
                "value_cos_heads": F.cosine_similarity(v[:, :1], v[:, 1:], dim=-1).cpu(),
                "contribution_heads": head_terms[:, 1:].norm(dim=-1).cpu(),
                "contribution": terms[1:].norm(dim=-1).cpu(),
                "cls_self_contribution": terms[0].norm().cpu(),
                "reconstruction_error": torch.tensor(error),
            })
        return hook

    def block_hook(self, index):
        def hook(module, inputs, output):
            output = output[0] if isinstance(output, tuple) else output
            tokens = output[0].detach().float()
            self.values[index]["token_cos"] = F.cosine_similarity(tokens[:1], tokens[1:], dim=-1).cpu()
        return hook


def pack(capture, grid):
    if len(capture.values) != len(capture.blocks):
        raise RuntimeError("Not all global blocks executed")
    records = [capture.values[index] for index in range(len(capture.blocks))]
    patch_count = grid[0] * grid[1]
    if any(record["contribution"].numel() != patch_count for record in records):
        raise RuntimeError("Patch count does not match the model grid")
    arrays = {key: torch.stack([record[key] for record in records]).numpy()
              for key in records[0]}
    for key in ("qnorm", "attention", "value_norm", "value_cos"):
        arrays[key] = arrays[key + "_heads"].mean(axis=1)
    arrays["grid"] = np.asarray(grid)
    return arrays


def draw_overlay(axis, rgb, values, grid, low, high, signed):
    axis.imshow(rgb)
    heat = axis.imshow(values.reshape(grid), cmap="coolwarm" if signed else "turbo",
                       vmin=low, vmax=high, interpolation="nearest", alpha=0.55,
                       extent=(-0.5, rgb.shape[1] - 0.5, rgb.shape[0] - 0.5, -0.5))
    axis.set_xticks([])
    axis.set_yticks([])
    return heat


def render(arrays, rgb, output_dir, per_head):
    layers = arrays["qnorm"].shape[0]
    grid = tuple(int(x) for x in arrays["grid"])
    fig, axes = plt.subplots(layers, len(PANELS) + 1, figsize=(18, max(4, layers * 2.35)), squeeze=False)
    for layer in range(layers):
        axes[layer, 0].imshow(rgb)
        axes[layer, 0].set_xticks([])
        axes[layer, 0].set_yticks([])
        axes[layer, 0].set_ylabel("Block %02d" % (layer + 1), fontsize=10)
    axes[0, 0].set_title("Night image")
    for column, (key, title, kind) in enumerate(PANELS, start=1):
        signed = kind == "signed"
        low = -1.0 if signed else 0.0
        high = 1.0 if signed else max(float(np.percentile(arrays[key], 99)), 1e-8)
        for layer in range(layers):
            heat = draw_overlay(axes[layer, column], rgb, arrays[key][layer], grid, low, high, signed)
        axes[0, column].set_title(title, fontsize=9)
        fig.colorbar(heat, ax=list(axes[:, column]), fraction=0.018, pad=0.02)
    fig.suptitle("Scale shared across layers within each column; cosines use [-1,1]\n"
                 "Attention update is observed, not final identity attribution", fontsize=11)
    fig.subplots_adjust(left=0.045, right=0.965, top=0.94, bottom=0.015, wspace=0.18, hspace=0.08)
    fig.savefig(str(output_dir / "all_layers.png"), dpi=120)
    plt.close(fig)

    if per_head:
        head_values = arrays["contribution_heads"]
        heads = head_values.shape[1]
        high = max(float(np.percentile(head_values, 99)), 1e-8)
        fig, axes = plt.subplots(layers, heads, figsize=(heads * 1.6, layers * 2.35), squeeze=False)
        for layer in range(layers):
            for head in range(heads):
                heat = draw_overlay(axes[layer, head], rgb, head_values[layer, head], grid, 0, high, False)
                if layer == 0:
                    axes[layer, head].set_title("Head %02d" % (head + 1), fontsize=8)
            axes[layer, 0].set_ylabel("Block %02d" % (layer + 1))
        fig.suptitle("Per-head contribution magnitudes | one scale across all heads/layers")
        fig.subplots_adjust(left=0.045, right=0.93, top=0.95, bottom=0.015, wspace=0.06, hspace=0.08)
        fig.colorbar(heat, ax=list(axes.flat), fraction=0.012, pad=0.015)
        fig.savefig(str(output_dir / "all_layers_per_head_contribution.png"), dpi=110)
        plt.close(fig)


def main():
    args = parse_args()
    indices = [int(item.strip()) for item in args.sample_indices.split(",")]
    if not indices or min(indices) < 0 or len(indices) != len(set(indices)):
        raise ValueError("sample_indices must be a nonempty set of nonnegative indices")
    from config import cfg as base_cfg
    from datasets import make_dataloader

    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.JPM = False
    # The complete teacher checkpoint is loaded below. Avoid reading an
    # unrelated pretrained backbone first (several minutes on this server).
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"checkpoint": args.teacher_checkpoint, "config": args.config_file,
                "indices": indices, "device": str(device), "grid": grid,
                "note": "Read-only visualization; maps do not establish identity-information gain.",
                "samples": []}
    rows = []
    capture = Capture(model.base)
    try:
        for index in indices:
            image, pid, camid, viewid, image_path = dataset[index]
            capture.clear()
            with torch.inference_mode():
                model(image.unsqueeze(0).to(device),
                      cam_label=torch.tensor([camid], device=device),
                      view_label=torch.tensor([viewid], device=device))
            arrays = pack(capture, grid)
            mean = torch.tensor(cfg.INPUT.PIXEL_MEAN).view(3, 1, 1)
            std = torch.tensor(cfg.INPUT.PIXEL_STD).view(3, 1, 1)
            rgb = (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            sample_dir = output_dir / ("sample_%05d" % index)
            sample_dir.mkdir()
            np.savez_compressed(str(sample_dir / "raw_layer_maps.npz"), **arrays)
            render(arrays, rgb, sample_dir, args.per_head)
            manifest["samples"].append({"index": index, "pid": int(pid),
                                        "camid": int(camid), "viewid": int(viewid),
                                        "image_path": str(image_path)})
            for layer in range(arrays["qnorm"].shape[0]):
                rows.append({"sample_index": index, "layer": layer + 1,
                             "cls_to_patch_attention_mass": float(arrays["attention"][layer].sum()),
                             "cls_self_attention_mass": float(arrays["cls_self_attention_heads"][layer].mean()),
                             "patch_contribution_norm_sum": float(arrays["contribution"][layer].sum()),
                             "mean_cls_patch_value_cosine": float(arrays["value_cos"][layer].mean()),
                             "reconstruction_error": float(arrays["reconstruction_error"][layer])})
            print("Saved %d layers for index %d -> %s" %
                  (len(capture.blocks), index, sample_dir), flush=True)
    finally:
        capture.close()
    with (output_dir / "layer_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Complete:", output_dir)


if __name__ == "__main__":
    main()
