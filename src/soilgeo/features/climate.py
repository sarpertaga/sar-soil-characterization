"""
Climate features (V3-F3).

Implements the Standardized Precipitation Index (SPI) — McKee et al. (1993) —
which the spec lists as the SPI-3 drought indicator. SPI is computed by:

  1. accumulating precipitation over a rolling window (3 months for SPI-3),
  2. fitting a gamma distribution to the accumulated series,
  3. transforming the gamma CDF to a standard-normal z-score.

Because precipitation can be exactly zero, the gamma fit uses the standard
zero-inflation correction: ``H(x) = q + (1 - q)·G(x)`` where ``q`` is the
probability of zero precipitation and ``G`` is the gamma CDF of the positive
values (Lloyd-Hughes & Saunders, 2002).

The download of CHIRPS / ERA5-Land grids (via ``cdsapi``) lives in
``soilgeo.acquisition.climate`` and requires credentials; the SPI math here is
pure ``scipy`` and unit-tested on synthetic series.
"""
import numpy as np
from scipy.stats import gamma, norm

from soilgeo.utils.logging import get_logger

log = get_logger(__name__)


def rolling_sum(series: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Trailing rolling sum over a 1-D series. Positions with fewer than
    ``window`` preceding samples are NaN, so the output length matches the
    input.
    """
    series = np.asarray(series, dtype=np.float64)
    csum = np.cumsum(series)
    out = np.full(series.shape, np.nan, dtype=np.float64)
    out[window - 1] = csum[window - 1]
    out[window:] = csum[window:] - csum[:-window]
    return out


def fit_gamma(values: np.ndarray) -> tuple[float, float, float]:
    """
    Fit a gamma distribution to strictly-positive precipitation accumulations.
    Returns ``(shape, loc, scale)`` with ``loc`` fixed at 0 (precip is
    non-negative and gamma is defined on x>0).
    """
    positive = values[values > 0]
    shape, loc, scale = gamma.fit(positive, floc=0.0)
    return float(shape), float(loc), float(scale)


def _zero_inflated_gamma_cdf(vals: np.ndarray, fit_vals: np.ndarray) -> np.ndarray:
    """H(x) = q + (1-q)·G(x) evaluated at ``vals``, fitted on ``fit_vals``."""
    q = float(np.mean(fit_vals == 0.0))
    shape, loc, scale_g = fit_gamma(fit_vals)
    return np.where(
        vals > 0,
        q + (1.0 - q) * gamma.cdf(vals, a=shape, loc=loc, scale=scale_g),
        q / 2.0,  # zero precip mapped to the lower tail mass
    )


def spi(precip_series: np.ndarray, scale: int = 3, start_month: int = 1) -> np.ndarray:
    """
    Standardized Precipitation Index at the given month-scale (SPI-3 by
    default). Returns a z-score array the same length as the input; the first
    ``scale-1`` entries are NaN (insufficient accumulation history).

    Following McKee et al. (1993), the gamma distribution is fitted **per
    calendar month** (the month each accumulation window ends in, derived from
    ``start_month`` = calendar month of the first sample). This makes SPI an
    anomaly relative to that month's own climatology rather than a measure of
    the seasonal cycle. Months with too few positive samples for a stable
    gamma fit (< 3) fall back to the all-months fit.
    """
    precip_series = np.asarray(precip_series, dtype=np.float64)
    acc = rolling_sum(precip_series, window=scale)

    valid = ~np.isnan(acc)
    vals = acc[valid]
    # Calendar month (1-12) in which each valid accumulation window ends.
    end_months = ((start_month - 1 + np.flatnonzero(valid)) % 12) + 1

    cdf = np.empty_like(vals)
    for month in np.unique(end_months):
        sel = end_months == month
        month_vals = vals[sel]
        fit_vals = month_vals if np.sum(month_vals > 0) >= 3 else vals
        cdf[sel] = _zero_inflated_gamma_cdf(month_vals, fit_vals)

    # Clip away from 0/1 so the inverse normal stays finite.
    cdf = np.clip(cdf, 1e-6, 1.0 - 1e-6)
    z = norm.ppf(cdf)

    out = np.full(precip_series.shape, np.nan, dtype=np.float64)
    out[valid] = z
    log.info("SPI-%d computed over %d samples (per-calendar-month fit)", scale, vals.size)
    return out


def spi_latest_raster(
    precip_stack: np.ndarray, scale: int = 3, start_month: int = 1
) -> np.ndarray:
    """
    Apply SPI-``scale`` per pixel to a monthly precipitation stack ``[T, H, W]``
    and return the **most recent** SPI value per pixel as ``[H, W]`` float32.
    ``start_month`` is the calendar month (1-12) of the stack's first layer.
    Pixels whose series is constant/degenerate are returned as NaN.
    """
    T, H, W = precip_stack.shape
    out = np.full((H, W), np.nan, dtype=np.float32)
    for i in range(H):
        for j in range(W):
            series = precip_stack[:, i, j]
            if not np.all(np.isfinite(series)) or np.nanstd(series) == 0:
                continue
            try:
                out[i, j] = spi(series, scale=scale, start_month=start_month)[-1]
            except Exception:  # noqa: BLE001 — degenerate gamma fit on a pixel
                continue
    return out
