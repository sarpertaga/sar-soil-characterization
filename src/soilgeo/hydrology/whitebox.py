"""Hydrological analysis via WhiteboxTools: flow accumulation, TWI, SPI."""
import subprocess
from pathlib import Path

import numpy as np
import rasterio

from soilgeo.utils.logging import get_logger

log = get_logger(__name__)

NODATA = -9999.0
# Minimum slope in degrees before tan(β) → 0 causes TWI → ∞
# Flat Konya plain has mean slope ~1°; floor prevents NaN on pits/flats
MIN_SLOPE_DEG = 0.1


def _get_wbt():
    import whitebox
    wbt = whitebox.WhiteboxTools()
    wbt.verbose = False
    return wbt


def compute_flow_accumulation(dem_path: Path, output_path: Path, work_dir: Path | None = None) -> Path:
    """Fill depressions then compute D8 specific contributing area."""
    if output_path.exists():
        log.info("Skipping flow_accumulation (exists)")
        return output_path

    work_dir = work_dir or output_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    filled = work_dir / "_dem_filled.tif"

    wbt = _get_wbt()
    wbt.work_dir = str(work_dir)
    wbt.fill_depressions(str(dem_path), str(filled))
    wbt.d8_flow_accumulation(str(filled), str(output_path), out_type="specific contributing area")
    log.info("Written: %s", output_path.name)
    return output_path


def compute_twi_spi(
    dem_path: Path,
    twi_output: Path,
    spi_output: Path,
    work_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Compute TWI = ln(a/tanβ) and SPI = a·tanβ with slope floor for flat terrain."""
    work_dir = work_dir or twi_output.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    fa_path = work_dir / "flow_acc.tif"
    compute_flow_accumulation(dem_path, fa_path, work_dir)

    if not twi_output.exists():
        log.info("Computing TWI (manual, slope floor=%.2f°)...", MIN_SLOPE_DEG)
        _compute_twi_manual(dem_path, fa_path, twi_output)

    if not spi_output.exists():
        log.info("Computing SPI...")
        _compute_spi(dem_path, fa_path, spi_output)

    return twi_output, spi_output


def _slope_array(dem_path: Path, work_dir: Path) -> np.ndarray:
    """Return slope in degrees as float64 array."""
    slope_path = work_dir / "_slope_twi.tif"
    if not slope_path.exists():
        subprocess.run(
            ["gdaldem", "slope", str(dem_path), str(slope_path), "-of", "GTiff"],
            check=True, capture_output=True,
        )
    with rasterio.open(slope_path) as src:
        return src.read(1).astype(np.float64)


def _compute_twi_manual(dem_path: Path, fa_path: Path, twi_output: Path) -> Path:
    """
    TWI = ln(a / tan(β))  — computed from flow accumulation + slope.
    Applies MIN_SLOPE_DEG floor so flat areas get finite (high) TWI
    instead of NaN/infinity. This is standard practice for plains.
    """
    with rasterio.open(fa_path) as src:
        fa = src.read(1).astype(np.float64)
        profile = src.profile
        nd_fa = src.nodata if src.nodata is not None else NODATA

    slope_deg = _slope_array(dem_path, fa_path.parent)

    mask = (fa == nd_fa) | np.isnan(fa) | (fa <= 0)
    # Apply slope floor: flat pixels get MIN_SLOPE_DEG, not 0
    slope_floored = np.clip(slope_deg, MIN_SLOPE_DEG, None)
    tan_slope = np.tan(np.radians(slope_floored))

    with np.errstate(divide="ignore", invalid="ignore"):
        twi = np.where(mask, NODATA, np.log(fa / tan_slope))

    twi_output.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", count=1, nodata=NODATA, compress="deflate")
    with rasterio.open(twi_output, "w", **profile) as dst:
        dst.write(twi.astype(np.float32), 1)

    valid = twi[twi != NODATA]
    log.info("TWI written: %s | range [%.2f, %.2f] mean=%.2f valid=%.1f%%",
             twi_output.name, valid.min(), valid.max(), valid.mean(),
             100 * valid.size / twi.size)
    return twi_output


def _compute_spi(dem_path: Path, fa_path: Path, spi_output: Path) -> Path:
    """SPI = a × tan(β)"""
    slope_deg = _slope_array(dem_path, fa_path.parent)

    with rasterio.open(fa_path) as src:
        fa = src.read(1).astype(np.float64)
        profile = src.profile
        nd_fa = src.nodata if src.nodata is not None else NODATA

    mask = (fa == nd_fa) | np.isnan(fa)
    slope_rad = np.radians(np.clip(slope_deg, MIN_SLOPE_DEG, None))
    spi = np.where(mask, NODATA, fa * np.tan(slope_rad))

    spi_output.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", count=1, nodata=NODATA, compress="deflate")
    with rasterio.open(spi_output, "w", **profile) as dst:
        dst.write(spi.astype(np.float32), 1)
    log.info("Written: %s", spi_output.name)
    return spi_output
