"""Hydrological analysis via WhiteboxTools: flow accumulation, TWI, SPI."""
import subprocess
from pathlib import Path

import numpy as np
import rasterio

from soilgeo.utils.logging import get_logger

log = get_logger(__name__)

NODATA = -9999.0


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
    """Compute TWI = ln(a/tanβ) and SPI = a·tanβ."""
    work_dir = work_dir or twi_output.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    fa_path = work_dir / "flow_acc.tif"
    compute_flow_accumulation(dem_path, fa_path, work_dir)

    if not twi_output.exists():
        log.info("Computing TWI...")
        wbt = _get_wbt()
        wbt.work_dir = str(work_dir)
        wbt.wetness_index(str(fa_path), str(dem_path), str(twi_output))
        log.info("Written: %s", twi_output.name)

    if not spi_output.exists():
        log.info("Computing SPI...")
        _compute_spi(dem_path, fa_path, spi_output)

    return twi_output, spi_output


def _compute_spi(dem_path: Path, fa_path: Path, spi_output: Path) -> Path:
    """SPI = a × tan(β) computed from flow_acc and slope arrays."""
    slope_path = spi_output.parent / "_slope_for_spi.tif"
    subprocess.run(
        ["gdaldem", "slope", str(dem_path), str(slope_path), "-of", "GTiff"],
        check=True, capture_output=True,
    )
    with rasterio.open(fa_path) as src:
        fa = src.read(1).astype(np.float64)
        profile = src.profile
        nd = src.nodata or NODATA

    with rasterio.open(slope_path) as src:
        slope_deg = src.read(1).astype(np.float64)

    slope_rad = np.radians(slope_deg)
    tan_slope = np.tan(np.clip(slope_rad, 1e-6, None))
    spi = fa * tan_slope

    mask = (fa == nd) | (slope_deg == NODATA)
    spi[mask] = NODATA

    spi_output.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", nodata=NODATA, compress="deflate")
    with rasterio.open(spi_output, "w", **profile) as dst:
        dst.write(spi.astype(np.float32), 1)
    log.info("Written: %s", spi_output.name)
    return spi_output
