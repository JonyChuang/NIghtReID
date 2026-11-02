#!/usr/bin/env python3
"""Periodic Q-Norm activation snapshots for TransReID training.

The monitor is deliberately independent from the training loop.  Import it in
``processor/processor.py`` (or ``train.py``), construct it once after the model
and train loader exist, call :meth:`on_epoch_end` after each epoch, and call
:meth:`close` in a ``finally`` block.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch

from visualize_qnorm import attention_map, find_attention, norm_map, save_panel, split_qkv


class QNormMonitor:
    """Save random Q-Norm panels every ``every`` epochs.

    ``loader`` is any TransReID-style loader yielding a tuple whose first item
    is an image tensor.  Camera and view labels are read from tuple positions 2
    and 3, matching the upstream TransReID train loader; ``view_index`` can be
    set to 4 for Night600 loaders that insert an extra camera field. Missing
    labels default to zero for models without SIE.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loader: Iterable,
        output_dir: str | Path,
        *,
        every: int = 10,
        num_images: int = 4,
        input_height: int = 256,
        input_width: int = 128,
        patch_size: int = 16,
        prefix_tokens: int = 1,
        layer: int = -1,
        camera_label: int = 0,
        view_label: int = 0,
        view_index: Optional[int] = None,
        alpha: float = 0.48,
        percentile: float = 99.0,
        device: Optional[torch.device | str] = None,
        save_npy: bool = False,
        pixel_mean: Optional[Iterable[float]] = None,
        pixel_std: Optional[Iterable[float]] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        if every <= 0 or num_images <= 0:
            raise ValueError("every and num_images must be positive")
        self.model = model.module if hasattr(model, "module") else model
        self.loader = loader
        self.output_dir = Path(output_dir)
        self.every = every
        self.num_images = num_images
        self.grid = (input_height // patch_size, input_width // patch_size)
        self.prefix_tokens = prefix_tokens
        self.camera_label = camera_label
        self.view_label = view_label
        self.view_index = view_index
        self.alpha = alpha
        self.percentile = percentile
        self.pixel_mean = torch.tensor(
            list(pixel_mean if pixel_mean is not None else (0.5, 0.5, 0.5)),
            dtype=torch.float32,
        )
        self.pixel_std = torch.tensor(
            list(pixel_std if pixel_std is not None else (0.5, 0.5, 0.5)),
            dtype=torch.float32,
        )
        self.device = torch.device(device) if device is not None else next(
            self.model.parameters()
        ).device
        self.save_npy = save_npy
        if enabled is None:
            enabled = (not torch.distributed.is_available() or
                       not torch.distributed.is_initialized() or
                       torch.distributed.get_rank() == 0)
        self.enabled = bool(enabled)
        self._captured: Dict[str, torch.Tensor] = {}
        self._closed = False

        self.attention_name, self.attention = find_attention(self.model, layer)
        self._hook = None

    @staticmethod
    def _first_batch_item(batch: Any) -> torch.Tensor:
        if isinstance(batch, dict):
            for key in ("img", "image", "images"):
                if key in batch:
                    return batch[key]
            raise RuntimeError("Loader dict has no image/img/images field")
        if isinstance(batch, (tuple, list)):
            if not batch:
                raise RuntimeError("Loader returned an empty tuple")
            return batch[0]
        return batch

    @staticmethod
    def _label(batch: Any, index: int, default: int, batch_size: int) -> torch.Tensor:
        value: Any = None
        if isinstance(batch, dict):
            keys = (("cam_label", "camera", "camid") if index == 2 else
                    ("view_label", "view", "viewid"))
            for key in keys:
                if key in batch:
                    value = batch[key]
                    break
        elif isinstance(batch, (tuple, list)) and len(batch) > index:
            value = batch[index]
        if value is None:
            return torch.full((batch_size,), default, dtype=torch.long)
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        value = value.reshape(-1).long()
        if value.numel() == 1 and batch_size > 1:
            value = value.expand(batch_size)
        return value[:batch_size]

    def _pre_hook(self, module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        x = inputs[0]
        with torch.no_grad():
            q, k, v = split_qkv(module, x)
            self._captured["q"] = q.float().detach()
            self._captured["k"] = k.float().detach()
            self._captured["v"] = v.float().detach()
            self._captured["attn"] = attention_map(
                q, k, module, self.prefix_tokens, self.grid
            ).detach()

    def _samples(self, epoch: int) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Collect a small pool and choose samples without consuming training state."""
        pool: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        # A bounded pool keeps an accidental huge dataset from stalling training.
        pool_limit = max(self.num_images * 8, self.num_images)
        for batch in self.loader:
            images = self._first_batch_item(batch)
            if not torch.is_tensor(images) or images.ndim < 4:
                raise RuntimeError("Loader image item must be a BCHW tensor")
            cams = self._label(batch, 2, self.camera_label, images.shape[0])
            # Upstream TransReID training tuples put view labels at index 3;
            # some Night600 validation tuples add a non-label field there and
            # put view IDs at index 4.  Prefer index 3 and fall back only when
            # it cannot be converted to numeric labels.
            if self.view_index is not None:
                views = self._label(batch, self.view_index, self.view_label, images.shape[0])
            else:
                try:
                    views = self._label(batch, 3, self.view_label, images.shape[0])
                except (TypeError, ValueError, RuntimeError):
                    views = self._label(batch, 4, self.view_label, images.shape[0])
            for image, cam, view in zip(images, cams, views):
                pool.append((image.detach().cpu(), cam.detach().cpu(), view.detach().cpu()))
                if len(pool) >= pool_limit:
                    break
            if len(pool) >= pool_limit:
                break
        if not pool:
            return []
        rng = random.Random(epoch)
        return rng.sample(pool, min(self.num_images, len(pool)))

    def _forward(self, image: torch.Tensor, cam: torch.Tensor, view: torch.Tensor) -> None:
        self._captured.clear()
        batch = image.unsqueeze(0).to(self.device, non_blocking=True)
        cam = cam.reshape(1).to(self.device, non_blocking=True).long()
        view = view.reshape(1).to(self.device, non_blocking=True).long()
        try:
            self.model(batch, cam_label=cam, view_label=view)
        except TypeError:
            self.model(batch)
        if "q" not in self._captured:
            raise RuntimeError("Q-Norm hook did not capture an attention input")

    def on_epoch_end(self, epoch: int) -> None:
        """Capture this epoch when it is a multiple of ``every``."""
        if self._closed or epoch <= 0 or epoch % self.every:
            return
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if distributed:
            torch.distributed.barrier()
            if torch.distributed.get_rank() != 0 or not self.enabled:
                torch.distributed.barrier()
                return
        elif not self.enabled:
            return
        try:
            samples = self._samples(epoch)
        except Exception:
            if distributed:
                torch.distributed.barrier()
            raise
        if not samples:
            print("[QNormMonitor] loader produced no samples")
            if distributed:
                torch.distributed.barrier()
            return
        states = {module: module.training for module in self.model.modules()}
        self.model.eval()
        epoch_dir = self.output_dir / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        # Install the hook only for these monitoring forwards. Keeping it
        # active during normal training would recompute qkv every batch.
        self._hook = self.attention.register_forward_pre_hook(self._pre_hook)
        try:
            with torch.no_grad():
                for index, (image, cam, view) in enumerate(samples):
                    self._forward(image, cam, view)
                    # Training transforms generally normalize tensors.  Undo the
                    # common normalization for a useful RGB background.
                    rgb = image.float().clone().cpu()
                    mean = self.pixel_mean[:, None, None]
                    std = self.pixel_std[:, None, None]
                    rgb = (rgb * std + mean).clamp(0, 1)
                    original = Image.fromarray((rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                    maps = {
                        "Q-Norm": norm_map(self._captured["q"], self.prefix_tokens, self.grid)[0],
                        "K-Norm": norm_map(self._captured["k"], self.prefix_tokens, self.grid)[0],
                        "V-Norm": norm_map(self._captured["v"], self.prefix_tokens, self.grid)[0],
                        "Attention": self._captured["attn"][0],
                    }
                    if self.save_npy:
                        raw = {name.lower().replace("-", "_"): value.cpu().numpy()
                               for name, value in maps.items()}
                        np.savez(epoch_dir / f"{index:02d}.npz", **raw)
                    save_panel(original, maps, epoch_dir / f"{index:02d}.png",
                               self.alpha, self.percentile)
            print(f"[QNormMonitor] saved {len(samples)} images to {epoch_dir} ({self.attention_name})")
        finally:
            self._hook.remove()
            self._hook = None
            for module, training in states.items():
                module.training = training
            if distributed:
                torch.distributed.barrier()

    def close(self) -> None:
        if not self._closed:
            if self._hook is not None:
                self._hook.remove()
                self._hook = None
            self._closed = True

    def __enter__(self) -> "QNormMonitor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
