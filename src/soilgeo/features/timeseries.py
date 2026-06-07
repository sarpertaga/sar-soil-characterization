"""
Per-pixel time-series features (V3-F1).

Reduces a co-registered ``[T, H, W]`` stack of one band (e.g. VV in dB, or an
optical index) over the temporal axis into a set of ``[H, W]`` summary rasters:
temporal mean, std, 10th/90th percentiles, seasonal amplitude, and a
wet-minus-dry seasonal delta.

NODATA is handled per-pixel along the time axis: a sample equal to ``nodata``
is excluded from that pixel's statistics. A pixel with no valid samples across
all T dates is written back as ``nodata`` in every output (never crashes).
"""
import warnings

import numpy as np

from soilgeo.utils.logging import get_logger

log = get_logger(__name__)

NODATA = -9999.0


def _to_nan(stack: np.ndarray, nodata: float) -> np.ndarray:
    """Float64 copy with NODATA replaced by NaN for nan-aware reductions."""
    arr = stack.astype(np.float64)
    arr[stack == nodata] = np.nan
    return arr


def _restore_nodata(result: np.ndarray, all_invalid: np.ndarray, nodata: float) -> np.ndarray:
    """Set fully-invalid pixels to nodata and cast to float32."""
    out = np.where(all_invalid, nodata, result)
    return out.astype(np.float32)


def compute_temporal_stats(stack: np.ndarray, nodata: float = NODATA) -> dict[str, np.ndarray]:
    """
    Compute per-pixel temporal statistics over a ``[T, H, W]`` stack.

    Returns a dict with float32 ``[H, W]`` arrays:
        mean, std, p10, p90, amplitude (= p90 - p10)
    Pixels with no valid samples are ``nodata`` in every output.
    """
    if stack.ndim != 3:
        raise ValueError(f"expected [T,H,W] stack, got shape {stack.shape}")

    arr = _to_nan(stack, nodata)
    all_invalid = np.all(np.isnan(arr), axis=0)

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # empty / all-NaN slices are handled by _restore_nodata below
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(arr, axis=0)
        # ddof=0 (population std), consistent with V2 std usage; constant -> 0
        std = np.nanstd(arr, axis=0)
        p10 = np.nanpercentile(arr, 10, axis=0)
        p90 = np.nanpercentile(arr, 90, axis=0)

    amplitude = p90 - p10

    stats = {
        "mean": _restore_nodata(mean, all_invalid, nodata),
        "std": _restore_nodata(std, all_invalid, nodata),
        "p10": _restore_nodata(p10, all_invalid, nodata),
        "p90": _restore_nodata(p90, all_invalid, nodata),
        "amplitude": _restore_nodata(amplitude, all_invalid, nodata),
    }
    log.info("Temporal stats over T=%d (%dx%d)", stack.shape[0], stack.shape[1], stack.shape[2])
    return stats


def compute_wet_dry_delta(
    stack: np.ndarray,
    wet_idx: list[int],
    dry_idx: list[int],
    nodata: float = NODATA,
) -> np.ndarray:
    """
    Wet-minus-dry seasonal delta: mean over wet-season dates minus mean over
    dry-season dates, per pixel. ``[H, W]`` float32, ``nodata`` where either
    season has no valid samples.
    """
    arr = _to_nan(stack, nodata)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        wet_mean = np.nanmean(arr[wet_idx], axis=0)
        dry_mean = np.nanmean(arr[dry_idx], axis=0)

    invalid = np.isnan(wet_mean) | np.isnan(dry_mean)
    delta = wet_mean - dry_mean
    return _restore_nodata(delta, invalid, nodata)
