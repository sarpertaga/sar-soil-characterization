"""Compute DEM-derived terrain layers via gdaldem."""
import subprocess
from pathlib import Path

import numpy as np
import rasterio

from soilgeo.utils.logging import get_logger

log = get_logger(__name__)


def _run_gdaldem(mode: str, input_path: Path, output_path: Path, extra_args: list[str] | None = None) -> Path:
    if output_path.exists():
        log.info("Skipping %s (exists): %s", mode, output_path.name)
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["gdaldem", mode, str(input_path), str(output_path),
           "-of", "GTiff", "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES"]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gdaldem {mode} failed: {result.stderr}")
    log.info("Written: %s", output_path.name)
    return output_path


def compute_slope(dem_path: Path, output_path: Path) -> Path:
    return _run_gdaldem("slope", dem_path, output_path)


def compute_aspect(dem_path: Path, output_path: Path) -> Path:
    return _run_gdaldem("aspect", dem_path, output_path)


def compute_roughness(dem_path: Path, output_path: Path) -> Path:
    return _run_gdaldem("roughness", dem_path, output_path)


def compute_hillshade(dem_path: Path, output_path: Path, azimuth: float = 315.0, altitude: float = 45.0) -> Path:
    return _run_gdaldem("hillshade", dem_path, output_path, ["-az", str(azimuth), "-alt", str(altitude)])


def compute_curvature(dem_path: Path, output_path: Path) -> Path:
    """Plan curvature via numpy finite differences."""
    if output_path.exists():
        log.info("Skipping curvature (exists): %s", output_path.name)
        return output_path
    with rasterio.open(dem_path) as src:
        elev = src.read(1).astype(np.float64)
        profile = src.profile
        res = src.res[0]

    zy, zx = np.gradient(elev, res)
    _, zxx = np.gradient(zx, res)
    zyy, _ = np.gradient(zy, res)
    curvature = -(zxx + zyy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(curvature.astype(np.float32), 1)
    log.info("Written: %s", output_path.name)
    return output_path
