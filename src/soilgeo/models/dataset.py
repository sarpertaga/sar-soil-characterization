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


class SoilTileDataset:
    """
    Torch ``Dataset`` of co-registered feature/label tiles for the U-Net.

    Given a ``[C, H, W]`` feature cube, a label array (``[H, W]`` for regression
    or integer classes), and a list of tile windows ``(row, col, h, w)``, each
    item is a normalized ``(x, y)`` pair. Features are standardized with the
    supplied :class:`NormStats` (computed on the train split only — V3-T2).
    During training, one random geometric augmentation is applied per item
    (V3-T3); the same transform hits both feature and label.

    Tiles smaller than the nominal size (edge tiles) are zero-padded to
    ``tile_size`` so they batch cleanly; padded label regions use ``pad_value``.
    """

    def __init__(
        self,
        cube: np.ndarray,
        label: np.ndarray,
        tiles: list,
        norm: NormStats,
        tile_size: int = 256,
        task: str = "regression",
        augment: bool = False,
        pad_value: float = NODATA,
    ):
        z = norm.normalize(cube)
        # Invalid feature pixels must not reach the network: after
        # standardization the raw NODATA sentinel (-9999) would dominate every
        # convolution and poison BatchNorm statistics. Zero is the per-channel
        # mean post-normalization, i.e. the neutral value.
        self.cube = np.where(z == NODATA, 0.0, z).astype(np.float32)
        self.label = label
        self.tiles = tiles
        self.tile_size = tile_size
        self.task = task
        self.augment = augment
        self.pad_value = pad_value

    def __len__(self):
        return len(self.tiles)

    def _pad(self, arr, t, fill):
        import numpy as _np
        if arr.ndim == 3:
            c, h, w = arr.shape
            out = _np.full((c, t, t), fill, dtype=arr.dtype)
            out[:, :h, :w] = arr
        else:
            h, w = arr.shape
            out = _np.full((t, t), fill, dtype=arr.dtype)
            out[:h, :w] = arr
        return out

    def __getitem__(self, idx):
        import random

        import torch

        r, c, h, w = self.tiles[idx]
        t = self.tile_size
        # Features padded with 0 (neutral post-normalization); labels padded
        # with pad_value (NODATA) so padded pixels stay masked out of the loss.
        x = self._pad(self.cube[:, r:r + h, c:c + w], t, 0.0)
        y = self._pad(self.label[r:r + h, c:c + w], t, self.pad_value)

        x = torch.from_numpy(x).float()
        if self.task == "regression":
            y = torch.from_numpy(y).float().unsqueeze(0)        # [1, t, t]
        else:
            y = torch.from_numpy(y).long()                       # [t, t]

        if self.augment:
            op = random.choice(["none", "hflip", "vflip", "rot90"])
            x, y = apply_augmentation(x, y, op)
        return x, y


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
