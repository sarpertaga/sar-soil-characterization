"""
Tile dataset, normalization, and augmentation for the U-Net (V3-T2, V3-T3).

* **Normalization (V3-T2):** per-channel mean/std are computed on the *training*
  split only and persisted with the model artifact via :class:`NormStats`
  (``save``/``load``). Inference re-uses the stored stats — no leakage from
  val/test.
* **Augmentation (V3-T3):** geometric only (horizontal/vertical flip, 90°
  rotation). These permute pixels without changing radiometry, so SAR
  incidence-angle semantics are preserved; the *same* transform is applied to
  both the feature tile and its label.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NODATA = -9999.0


@dataclass
class NormStats:
    """Per-channel normalization statistics, saved alongside the model."""

    mean: np.ndarray  # shape (C,)
    std: np.ndarray   # shape (C,)

    def normalize(self, cube: np.ndarray) -> np.ndarray:
        """Standardize a ``[C, H, W]`` cube; nodata pixels are left untouched."""
        m = self.mean.reshape(-1, 1, 1)
        s = self.std.reshape(-1, 1, 1)
        valid = cube != NODATA
        z = (cube - m) / s
        return np.where(valid, z, NODATA).astype(np.float32)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"mean": self.mean.tolist(), "std": self.std.tolist()}
        path.write_text(json.dumps(payload, indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> NormStats:
        d = json.loads(Path(path).read_text())
        return cls(mean=np.asarray(d["mean"], dtype=np.float64),
                   std=np.asarray(d["std"], dtype=np.float64))


def compute_norm_stats(train_cubes: list[np.ndarray], nodata: float = NODATA) -> NormStats:
    """
    Compute per-channel mean/std over the **training** cubes only, ignoring
    nodata. Each cube is ``[C, H, W]``; all cubes must share C.
    """
    n_ch = train_cubes[0].shape[0]
    sums = np.zeros(n_ch, dtype=np.float64)
    sqs = np.zeros(n_ch, dtype=np.float64)
    counts = np.zeros(n_ch, dtype=np.float64)
    for cube in train_cubes:
        for c in range(n_ch):
            vals = cube[c][cube[c] != nodata]
            sums[c] += vals.sum()
            sqs[c] += np.square(vals.astype(np.float64)).sum()
            counts[c] += vals.size
    mean = sums / np.maximum(counts, 1)
    var = sqs / np.maximum(counts, 1) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    return NormStats(mean=mean, std=std)


def apply_augmentation(x, y, op: str):
    """
    Apply one geometric augmentation to a feature tile ``x`` (``[C,H,W]``) and
    its label ``y`` (``[H,W]`` or ``[C,H,W]``). Supported ops: ``hflip``,
    ``vflip``, ``rot90``, ``none``. The same transform is applied to both.
    """
    import torch  # local import: dataset is only used in the DL env

    if op == "none":
        return x, y
    if op == "hflip":
        return torch.flip(x, dims=[-1]), torch.flip(y, dims=[-1])
    if op == "vflip":
        return torch.flip(x, dims=[-2]), torch.flip(y, dims=[-2])
    if op == "rot90":
        return torch.rot90(x, k=1, dims=[-2, -1]), torch.rot90(y, k=1, dims=[-2, -1])
    raise ValueError(f"unknown augmentation op: {op!r}")
