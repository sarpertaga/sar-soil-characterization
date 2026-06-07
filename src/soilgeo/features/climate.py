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


def spi(precip_series: np.ndarray, scale: int = 3) -> np.ndarray:
    """
    Standardized Precipitation Index at the given month-scale (SPI-3 by
    default). Returns a z-score array the same length as the input; the first
    ``scale-1`` entries are NaN (insufficient accumulation history).
    """
    precip_series = np.asarray(precip_series, dtype=np.float64)
    acc = rolling_sum(precip_series, window=scale)

    valid = ~np.isnan(acc)
    vals = acc[valid]

    # Zero-inflation: probability of exactly-zero accumulation.
    q = float(np.mean(vals == 0.0))
    shape, loc, scale_g = fit_gamma(vals)

    cdf = np.where(
        vals > 0,
        q + (1.0 - q) * gamma.cdf(vals, a=shape, loc=loc, scale=scale_g),
        q / 2.0,  # zero precip mapped to the lower tail mass
    )
    # Clip away from 0/1 so the inverse normal stays finite.
    cdf = np.clip(cdf, 1e-6, 1.0 - 1e-6)
    z = norm.ppf(cdf)

    out = np.full(precip_series.shape, np.nan, dtype=np.float64)
    out[valid] = z
    log.info("SPI-%d computed over %d samples", scale, vals.size)
    return out
