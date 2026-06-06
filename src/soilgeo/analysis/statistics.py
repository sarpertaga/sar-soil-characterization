"""Stratified statistics: Kruskal-Wallis, Spearman, quantile stratification."""
from typing import Sequence

import numpy as np
from scipy import stats

from soilgeo.utils.logging import get_logger

log = get_logger(__name__)


def stratify_by_quantiles(values: np.ndarray, n_quantiles: int = 4) -> np.ndarray:
    """Assign 0-indexed quantile class labels to a 1D array."""
    quantile_edges = np.quantile(values, np.linspace(0, 1, n_quantiles + 1))
    labels = np.digitize(values, quantile_edges[1:-1])
    return labels.astype(np.uint8)


def kruskal_wallis_test(groups: Sequence[np.ndarray]) -> dict:
    """Kruskal-Wallis H test on two or more groups."""
    stat, p = stats.kruskal(*groups)
    result = {"statistic": float(stat), "p_value": float(p)}
    log.info("Kruskal-Wallis H=%.3f p=%.4g", stat, p)
    return result


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> dict:
    """Spearman rank correlation between two arrays."""
    rho, p = stats.spearmanr(x, y)
    result = {"rho": float(rho), "p_value": float(p)}
    log.info("Spearman ρ=%.4f p=%.4g", rho, p)
    return result


def backscatter_stats_by_stratum(
    backscatter: np.ndarray,
    strata: np.ndarray,
    nodata: float = -9999.0,
) -> dict:
    """Per-stratum backscatter statistics."""
    results = {}
    for label in np.unique(strata):
        mask = (strata == label) & (backscatter != nodata)
        vals = backscatter[mask]
        if vals.size == 0:
            continue
        results[int(label)] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75)),
            "n": int(vals.size),
        }
    return results
