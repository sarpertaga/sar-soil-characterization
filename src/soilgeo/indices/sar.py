"""SAR indices: VV/VH ratio and SAR Moisture Index."""
from pathlib import Path

import numpy as np
import rasterio

from soilgeo.utils.logging import get_logger

log = get_logger(__name__)
NODATA = -9999.0


def compute_vv_vh_ratio(vv_path: Path, vh_path: Path, output_path: Path) -> Path:
    """VV/VH ratio in dB = VV_dB − VH_dB."""
    if output_path.exists():
        log.info("Skipping vv_vh_ratio (exists)")
        return output_path

    with rasterio.open(vv_path) as src:
        vv = src.read(1).astype(np.float32)
        profile = src.profile
        nd = src.nodata or NODATA
    with rasterio.open(vh_path) as src:
        vh = src.read(1).astype(np.float32)

    mask = (vv == nd) | (vh == nd)
    ratio = np.where(mask, NODATA, vv - vh)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", count=1, nodata=NODATA, compress="deflate")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(ratio, 1)
    log.info("Written: %s", output_path.name)
    return output_path


def compute_nddi(
    wet_path: Path,
    dry_path: Path,
    output_path: Path,
) -> Path:
    """
    Normalized Difference Dryness Index (NDDI) from two VV composites.
    Computed in linear power scale to be physically meaningful:

        σ_linear = 10^(VV_dB / 10)
        NDDI = (σ_wet − σ_dry) / (σ_wet + σ_dry)   ∈ [-1, 1]

    Positive → wet season has higher backscatter (soil absorbed water).
    Negative → dry season has higher backscatter (bare soil, specular).
    ~0       → no seasonal difference.
    """
    if output_path.exists():
        log.info("Skipping nddi (exists)")
        return output_path

    with rasterio.open(wet_path) as src:
        vv_wet_db = src.read(1).astype(np.float32)
        profile = src.profile
        nd = src.nodata or NODATA
    with rasterio.open(dry_path) as src:
        vv_dry_db = src.read(1).astype(np.float32)
        nd_dry = src.nodata or NODATA

    mask = (vv_wet_db == nd) | (vv_dry_db == nd_dry)

    # dB → linear power
    wet_lin = np.where(mask, np.nan, 10.0 ** (vv_wet_db / 10.0))
    dry_lin = np.where(mask, np.nan, 10.0 ** (vv_dry_db / 10.0))

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = wet_lin + dry_lin
        nddi = np.where(
            mask | (denom < 1e-12),
            NODATA,
            (wet_lin - dry_lin) / denom,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", count=1, nodata=NODATA, compress="deflate")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(nddi.astype(np.float32), 1)

    valid = nddi[nddi != NODATA]
    log.info(
        "NDDI written: %s | range [%.3f, %.3f] mean=%.3f",
        output_path.name,
        float(np.nanmin(valid)), float(np.nanmax(valid)), float(np.nanmean(valid)),
    )
    return output_path
