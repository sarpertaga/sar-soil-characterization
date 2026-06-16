"""
Feature stack builder for soil property modelling.

Training strategy (250 m → 10 m downscaling):
  1. All 10 m feature rasters are aggregated to the SoilGrids 250 m grid.
  2. A RF model is trained on (n_sg_pixels, n_features) → soil property.
  3. The same model is applied at 10 m to produce high-resolution predictions.

This gives physically meaningful 10 m soil maps rather than bilinearly
upsampled SoilGrids products.
"""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from soilgeo.utils.logging import get_logger

log = get_logger(__name__)

NODATA = -9999.0

FEATURE_NAMES = [
    # SAR
    "vv_wet", "vh_wet", "vv_dry", "vh_dry", "nddi",
    # Terrain / hydrology
    "slope", "twi", "curvature",
    # S2 optical indices
    "bsi", "clay_index", "ndvi", "ndwi", "iron_oxide",
]


def _load_band(path: Path, band: int = 1) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        return src.read(band).astype(np.float32), src.profile.copy()


def _aggregate_to_grid(
    src_path: Path,
    ref_profile: dict,
    ref_shape: tuple[int, int],
) -> np.ndarray:
    """Reproject + mean-aggregate a 10 m raster to a coarser reference grid."""
    out = np.full(ref_shape, NODATA, dtype=np.float32)
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=Resampling.average,
            src_nodata=NODATA,
            dst_nodata=NODATA,
        )
    return out


def build_training_matrix(
    feature_paths: dict[str, Path],
    target_paths: dict[str, Path],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    """
    Build training feature matrix at the SoilGrids (coarse) resolution.

    Args:
        feature_paths: mapping of feature_name → 10 m raster path
        target_paths:  mapping of target_name → SoilGrids raster path (reference grid)

    Returns:
        X:            float32 array (n_valid_pixels, n_features)
        y_dict:       dict of target_name → float32 array (n_valid_pixels,)
        feat_names:   ordered list of feature names in X columns
    """
    # Use first target raster as reference grid for training
    ref_path = next(iter(target_paths.values()))
    with rasterio.open(ref_path) as ref:
        ref_profile = ref.profile.copy()
        ref_shape = (ref.height, ref.width)

    log.info("Reference grid: %dx%d px (%.0f m)",
             ref_shape[1], ref_shape[0],
             abs(ref_profile["transform"].a))

    # Aggregate features to reference resolution
    feat_names = []
    feat_arrays = []
    for name, path in feature_paths.items():
        arr = _aggregate_to_grid(path, ref_profile, ref_shape)
        feat_arrays.append(arr.flatten())
        feat_names.append(name)
        log.info("  feature %s: valid=%.1f%%", name,
                 100 * np.sum(arr != NODATA) / arr.size)

    # Load targets
    y_flat: dict[str, np.ndarray] = {}
    for name, path in target_paths.items():
        arr, _ = _load_band(path)
        # If target resolution doesn't match reference, resample
        if arr.shape != ref_shape:
            tmp = np.full(ref_shape, NODATA, dtype=np.float32)
            with rasterio.open(path) as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=tmp,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_profile["transform"],
                    dst_crs=ref_profile["crs"],
                    resampling=Resampling.bilinear,
                    src_nodata=NODATA,
                    dst_nodata=NODATA,
                )
            arr = tmp
        y_flat[name] = arr.flatten()

    # Valid mask: pixel must have all features and all targets
    X_raw = np.column_stack(feat_arrays)
    valid = np.all(X_raw != NODATA, axis=1)
    for y in y_flat.values():
        valid &= (y != NODATA) & np.isfinite(y)

    log.info("Training pixels: %d / %d (%.1f%%)", valid.sum(), valid.size,
             100 * valid.sum() / valid.size)

    X = X_raw[valid].astype(np.float32)
    y_dict = {name: arr[valid].astype(np.float32) for name, arr in y_flat.items()}
    return X, y_dict, feat_names


def build_prediction_stack(
    feature_paths: dict[str, Path],
) -> tuple[np.ndarray, dict, tuple[int, int]]:
    """
    Load all feature rasters at full (10 m) resolution for prediction.

    Returns:
        X_full:   float32 array (n_pixels, n_features)
        profile:  rasterio profile of the reference raster
        shape:    (height, width) for reshaping predictions
    """
    # Use first feature as reference for grid alignment
    ref_path = next(iter(feature_paths.values()))
    with rasterio.open(ref_path) as ref:
        ref_profile = ref.profile.copy()
        ref_shape = (ref.height, ref.width)

    feat_arrays = []
    for name, path in feature_paths.items():
        with rasterio.open(path) as src:
            if (src.height, src.width) != ref_shape or src.crs != ref_profile["crs"]:
                arr = np.full(ref_shape, NODATA, dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=arr,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_profile["transform"],
                    dst_crs=ref_profile["crs"],
                    resampling=Resampling.bilinear,
                    src_nodata=NODATA,
                    dst_nodata=NODATA,
                )
            else:
                arr = src.read(1).astype(np.float32)
        feat_arrays.append(arr.flatten())

    X_full = np.column_stack(feat_arrays).astype(np.float32)
    return X_full, ref_profile, ref_shape
