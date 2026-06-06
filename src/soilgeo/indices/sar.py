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


def compute_moisture_index(
    scene_path: Path,
    dry_path: Path,
    wet_path: Path,
    output_path: Path,
) -> Path:
    """
    SAR Moisture Index: MI = (σ_t − σ_dry) / (σ_wet − σ_dry)
    Output clamped to [0, 1].
    """
    if output_path.exists():
        log.info("Skipping moisture_index (exists)")
        return output_path

    with rasterio.open(scene_path) as src:
        sigma_t = src.read(1).astype(np.float32)
        profile = src.profile
        nd = src.nodata or NODATA
    with rasterio.open(dry_path) as src:
        sigma_dry = src.read(1).astype(np.float32)
        nd_dry = src.nodata or NODATA
    with rasterio.open(wet_path) as src:
        sigma_wet = src.read(1).astype(np.float32)
        nd_wet = src.nodata or NODATA

    mask = (sigma_t == nd) | (sigma_dry == nd_dry) | (sigma_wet == nd_wet)
    denom = sigma_wet - sigma_dry

    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.where(
            mask | (np.abs(denom) < 1e-6),
            NODATA,
            np.clip((sigma_t - sigma_dry) / denom, 0.0, 1.0),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", count=1, nodata=NODATA, compress="deflate")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mi.astype(np.float32), 1)
    log.info("Moisture index written: %s", output_path.name)
    return output_path
