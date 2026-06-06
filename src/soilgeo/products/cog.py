"""Product catalog and Construction Moisture Risk Index."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

from soilgeo.utils.logging import get_logger

log = get_logger(__name__)
NODATA = -9999.0


def write_product_catalog(products_dir: Path, entries: list[dict]) -> Path:
    catalog_path = products_dir / "catalog.json"
    catalog = {
        "_generated_at": datetime.now(timezone.utc).isoformat(),
        "products": entries,
    }
    catalog_path.write_text(json.dumps(catalog, indent=2))
    log.info("Product catalog written: %d entries", len(entries))
    return catalog_path


def compute_construction_risk(
    mi_path: Path,
    flow_acc_path: Path,
    slope_path: Path,
    output_path: Path,
    weights: dict | None = None,
) -> Path:
    """
    Construction Moisture Risk Index:
    CRI = w_mi × MI + w_fa × norm(log(FA+1)) + w_slope × (1 − slope/max_slope)
    Output: [0, 1] float32 COG.
    """
    if output_path.exists():
        log.info("Skipping construction_risk (exists)")
        return output_path

    weights = weights or {"moisture_index": 0.5, "flow_accumulation_log": 0.3, "low_slope_factor": 0.2}

    with rasterio.open(mi_path) as src:
        mi = src.read(1).astype(np.float32)
        profile = src.profile
        nd_mi = src.nodata or NODATA
    with rasterio.open(flow_acc_path) as src:
        fa = src.read(1).astype(np.float64)
        nd_fa = src.nodata or NODATA
    with rasterio.open(slope_path) as src:
        slope = src.read(1).astype(np.float32)
        nd_sl = src.nodata or NODATA

    mask = (mi == nd_mi) | (fa == nd_fa) | (slope == nd_sl)

    fa_log = np.log1p(np.where(fa == nd_fa, 0.0, fa))
    fa_norm = fa_log / (fa_log[~mask].max() + 1e-9)
    slope_max = slope[~mask].max()
    low_slope = np.clip(1.0 - slope / (slope_max + 1e-9), 0.0, 1.0)
    mi_clipped = np.clip(mi, 0.0, 1.0)

    cri = (
        weights["moisture_index"] * mi_clipped
        + weights["flow_accumulation_log"] * fa_norm.astype(np.float32)
        + weights["low_slope_factor"] * low_slope
    )
    cri[mask] = NODATA

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", count=1, nodata=NODATA, compress="deflate")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(cri.astype(np.float32), 1)
    log.info("Construction Risk Index written: %s", output_path.name)
    return output_path
