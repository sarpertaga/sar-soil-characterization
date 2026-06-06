"""Run stratified statistical analysis and write results JSON."""
import json
from pathlib import Path

import numpy as np
import rasterio

from soilgeo.analysis.statistics import (
    stratify_by_quantiles,
    kruskal_wallis_test,
    spearman_correlation,
    backscatter_stats_by_stratum,
)
from soilgeo.utils.logging import get_logger

log = get_logger(__name__)
NODATA = -9999.0


def run_stats_analysis(
    vv_wet_path: Path,
    vv_dry_path: Path,
    nddi_path: Path,
    twi_path: Path,
    slope_path: Path,
    output_path: Path,
) -> dict:
    """
    Stratified statistical analysis (V1-F6):
    - Kruskal-Wallis: backscatter distributions by TWI quantile + slope class
    - Spearman: NDDI vs TWI, NDDI vs slope, VV_wet vs VV_dry
    Returns results dict and saves to output_path JSON.
    """
    def _read(p):
        with rasterio.open(p) as src:
            d = src.read(1).astype(np.float32)
            nd = src.nodata if src.nodata is not None else NODATA
        d[d == nd] = np.nan
        return d.flatten()

    vv_wet  = _read(vv_wet_path)
    vv_dry  = _read(vv_dry_path)
    nddi    = _read(nddi_path)
    twi     = _read(twi_path)
    slope   = _read(slope_path)

    # Align sizes (resample may produce slightly different dims)
    min_size = min(vv_wet.size, nddi.size, twi.size, slope.size)
    vv_wet = vv_wet[:min_size]
    vv_dry = vv_dry[:min_size]
    nddi   = nddi[:min_size]
    twi    = twi[:min_size]
    slope  = slope[:min_size]

    # Common valid mask
    valid = ~(np.isnan(vv_wet) | np.isnan(nddi) | np.isnan(twi) | np.isnan(slope))
    vv_wet_v = vv_wet[valid]
    vv_dry_v = vv_dry[valid]
    nddi_v   = nddi[valid]
    twi_v    = twi[valid]
    slope_v  = slope[valid]

    log.info("Valid pixels for stats: %d (%.1f%%)", valid.sum(), 100*valid.mean())

    results = {}

    # ── Spearman correlations ────────────────────────────────────────────────
    log.info("Computing Spearman correlations...")
    results["spearman"] = {
        "nddi_vs_twi":    spearman_correlation(nddi_v, twi_v),
        "nddi_vs_slope":  spearman_correlation(nddi_v, slope_v),
        "vv_wet_vs_dry":  spearman_correlation(vv_wet_v, vv_dry_v),
        "vv_wet_vs_twi":  spearman_correlation(vv_wet_v, twi_v),
    }

    # ── Kruskal-Wallis: VV backscatter by TWI quantile ───────────────────────
    log.info("Kruskal-Wallis: VV by TWI quantile...")
    twi_strata = stratify_by_quantiles(twi_v, n_quantiles=4)
    kw_groups_wet = [vv_wet_v[twi_strata == q] for q in range(4)]
    kw_groups_dry = [vv_dry_v[twi_strata == q] for q in range(4)]
    results["kruskal_wallis"] = {
        "vv_wet_by_twi_quantile": kruskal_wallis_test(kw_groups_wet),
        "vv_dry_by_twi_quantile": kruskal_wallis_test(kw_groups_dry),
        "nddi_by_twi_quantile":   kruskal_wallis_test(
            [nddi_v[twi_strata == q] for q in range(4)]
        ),
    }

    # ── Kruskal-Wallis: VV by slope class ────────────────────────────────────
    slope_strata = stratify_by_quantiles(slope_v, n_quantiles=4)
    results["kruskal_wallis"]["vv_wet_by_slope"] = kruskal_wallis_test(
        [vv_wet_v[slope_strata == q] for q in range(4)]
    )

    # ── Backscatter stats by TWI stratum ─────────────────────────────────────
    results["backscatter_by_twi"] = {
        "wet": backscatter_stats_by_stratum(vv_wet_v, twi_strata),
        "dry": backscatter_stats_by_stratum(vv_dry_v, twi_strata),
    }

    # ── NDDI summary ─────────────────────────────────────────────────────────
    results["nddi_summary"] = {
        "mean": float(np.mean(nddi_v)),
        "std":  float(np.std(nddi_v)),
        "p10":  float(np.percentile(nddi_v, 10)),
        "p90":  float(np.percentile(nddi_v, 90)),
        "pct_positive": float((nddi_v > 0.05).mean() * 100),
        "pct_negative": float((nddi_v < -0.05).mean() * 100),
        "pct_neutral":  float((np.abs(nddi_v) <= 0.05).mean() * 100),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    log.info("Stats written: %s", output_path.name)
    return results
