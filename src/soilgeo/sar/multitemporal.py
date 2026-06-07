"""
Multi-temporal speckle filtering (V3-F2) — Quegan & Yu (2001).

Replaces V1's single-scene Refined Lee for the time-series SAR stack. Given a
set of ``T`` co-registered intensity images, the filtered value of image ``k``
at pixel ``p`` is::

    Î_k(p) = m_k(p) · (1/T) Σ_j [ I_j(p) / m_j(p) ]

where ``m_j`` is the local spatial mean of image ``j`` in a moving window. The
temporal averaging of the normalized ratios suppresses speckle while the
``m_k`` prefactor preserves each date's radiometric level. The estimator
operates on **linear intensity** (σ⁰ in power), so dB stacks must be converted
first with :func:`db_to_linear` and back with :func:`linear_to_db`.

Reference: Quegan, S. & Yu, J.J. (2001). *Filtering of multichannel SAR
images.* IEEE TGRS, 39(11).
"""
import numpy as np
from scipy.ndimage import uniform_filter

from soilgeo.utils.logging import get_logger

log = get_logger(__name__)

NODATA = -9999.0


def db_to_linear(db: np.ndarray, nodata: float = NODATA) -> np.ndarray:
    """dB -> linear power, preserving nodata."""
    out = np.where(db == nodata, nodata, 10.0 ** (db / 10.0))
    return out.astype(np.float32)


def linear_to_db(lin: np.ndarray, nodata: float = NODATA) -> np.ndarray:
    """Linear power -> dB, preserving nodata and guarding non-positive values."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(
            (lin == nodata) | (lin <= 0),
            nodata,
            10.0 * np.log10(lin),
        )
    return out.astype(np.float32)


def _local_mean(img: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    """
    NODATA-aware local spatial mean over a square window.

    Computed as box-sum(valid values) / box-sum(valid count) so that masked
    pixels neither contribute to nor bias the neighbourhood mean.
    """
    filled = np.where(valid, img, 0.0).astype(np.float64)
    count = valid.astype(np.float64)
    sum_ = uniform_filter(filled, size=window, mode="nearest") * (window * window)
    cnt = uniform_filter(count, size=window, mode="nearest") * (window * window)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(cnt > 0, sum_ / cnt, np.nan)
    return mean


def quegan_yu_filter(
    stack_linear: np.ndarray,
    window: int = 7,
    nodata: float = NODATA,
) -> np.ndarray:
    """
    Apply the Quegan & Yu multi-temporal filter to a ``[T, H, W]`` linear
    intensity stack. Returns a float32 stack of the same shape; pixels that are
    nodata in the input remain nodata in the output.
    """
    if stack_linear.ndim != 3:
        raise ValueError(f"expected [T,H,W] stack, got shape {stack_linear.shape}")

    T = stack_linear.shape[0]
    arr = stack_linear.astype(np.float64)
    valid = stack_linear != nodata

    # Local spatial mean m_j for each date, and the normalized ratio I_j / m_j.
    means = np.empty_like(arr)
    ratios = np.zeros_like(arr)
    ratio_valid = np.zeros_like(arr)
    for j in range(T):
        mj = _local_mean(arr[j], valid[j], window)
        means[j] = mj
        good = valid[j] & np.isfinite(mj) & (mj > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios[j] = np.where(good, arr[j] / mj, 0.0)
        ratio_valid[j] = good.astype(np.float64)

    # Temporal average of the valid ratios at each pixel.
    cnt = ratio_valid.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_bar = np.where(cnt > 0, ratios.sum(axis=0) / cnt, np.nan)

    out = np.empty_like(arr)
    for j in range(T):
        filtered = means[j] * ratio_bar
        out[j] = np.where(valid[j] & np.isfinite(filtered), filtered, nodata)

    log.info("Quegan-Yu filter applied: T=%d window=%d", T, window)
    return out.astype(np.float32)
