"""
Feature-cube assembly for V3 (build_cube stage).

Co-registers every V3 feature raster (SAR time-series statistics, S2 seasonal
NDVI/NDMI, terrain, hydrology, climate) and the SoilGrids label rasters onto a
common 10 m analysis grid, then produces:

* a ``[C, H, W]`` feature cube (optionally persisted to Zarr for the U-Net),
* a per-pixel feature matrix ``X`` + per-target label vectors ``y`` for the GBM,
* a **spatial-block group id** per sample so GroupKFold / tiling can keep blocks
  intact (risk R4).

Labels come from 250 m SoilGrids resampled to 10 m — the explicit label
super-resolution setup (V3-T5); downstream evaluation aggregates back to 250 m.
"""
from pathlib import Path

import numpy as np
import rasterio

from soilgeo.utils.geo import resample_to_match
from soilgeo.utils.logging import get_logger

log = get_logger(__name__)

NODATA = -9999.0


def assemble_cube(
    feature_paths: dict[str, Path],
    ref_path: Path,
    work_dir: Path,
) -> tuple[np.ndarray, list[str], dict]:
    """
    Resample every feature raster to the reference grid and stack into a
    ``[C, H, W]`` float32 cube. Returns ``(cube, feature_names, profile)``.
    ``feature_names`` is sorted for deterministic channel ordering.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
        H, W = ref.height, ref.width

    names = sorted(feature_paths)
    bands = []
    for name in names:
        aligned = work_dir / f"_aligned_{name}.tif"
        resample_to_match(feature_paths[name], ref_path, aligned)
        with rasterio.open(aligned) as src:
            bands.append(src.read(1).astype(np.float32))
    cube = np.stack(bands)  # [C, H, W]
    log.info("Feature cube assembled: C=%d H=%d W=%d", cube.shape[0], H, W)
    profile.update(count=cube.shape[0], dtype="float32", nodata=NODATA)
    return cube, names, profile


def build_matrix(
    cube: np.ndarray,
    label_paths: dict[str, Path],
    ref_path: Path,
    work_dir: Path,
    block_px: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Build the per-pixel training matrix from a feature cube + label rasters.

    Returns:
        X:        ``[N, C]`` features for valid samples
        y_dict:   ``{target: [N]}`` label vectors (resampled to the grid)
        groups:   ``[N]`` spatial-block id per sample (for GroupKFold / tiles)
        valid:    ``[H, W]`` boolean mask of which pixels became samples
    """
    C, H, W = cube.shape

    labels = {}
    for target, path in label_paths.items():
        aligned = work_dir / f"_label_{target}.tif"
        resample_to_match(path, ref_path, aligned)
        with rasterio.open(aligned) as src:
            labels[target] = src.read(1).astype(np.float32)

    feat_valid = np.all(cube != NODATA, axis=0)            # [H, W]
    label_valid = np.logical_and.reduce([labels[t] != NODATA for t in labels])
    valid = feat_valid & label_valid

    rows, cols = np.where(valid)
    X = cube[:, rows, cols].T                              # [N, C]
    y_dict = {t: labels[t][rows, cols] for t in labels}

    n_block_cols = (W // block_px) + 1
    groups = (rows // block_px) * n_block_cols + (cols // block_px)

    log.info("Training matrix: N=%d samples, C=%d feats, %d spatial blocks",
             X.shape[0], C, len(np.unique(groups)))
    return X, y_dict, groups, valid


def save_cube_zarr(cube: np.ndarray, feature_names: list[str], path: Path) -> Path:
    """Persist the feature cube to Zarr with channel names (for the U-Net)."""
    import zarr

    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = (cube.shape[0], min(256, cube.shape[1]), min(256, cube.shape[2]))
    root = zarr.open_group(str(path), mode="w")
    # zarr 3.x uses create_array(shape=...); fall back to 2.x create_dataset.
    try:
        arr = root.create_array("cube", shape=cube.shape, chunks=chunks, dtype=cube.dtype)
        arr[:] = cube
    except (AttributeError, TypeError):
        root.create_dataset("cube", data=cube, chunks=chunks, overwrite=True)
    root.attrs["feature_names"] = list(feature_names)
    log.info("Feature cube saved to Zarr: %s %s", path.name, cube.shape)
    return path
