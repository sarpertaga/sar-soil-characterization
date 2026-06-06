"""Download Copernicus DEM GLO-30 tiles from AWS Open Data (no credentials required)."""
import math
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling

from soilgeo.utils.logging import get_logger

log = get_logger(__name__)

_S3_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def build_dem_tile_ids(west: float, south: float, east: float, north: float) -> list[str]:
    """Return GLO-30 tile IDs (1°×1°) covering the bounding box."""
    tile_ids = []
    for lat in range(math.floor(south), math.ceil(north)):
        for lon in range(math.floor(west), math.ceil(east)):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            tile_ids.append(f"{ns}{abs(lat):02d}_{ew}{abs(lon):03d}")
    return tile_ids


def _tile_url(tile_id: str) -> str:
    folder = f"Copernicus_DSM_COG_10_{tile_id}_00_DEM"
    return f"{_S3_BASE}/{folder}/{folder}.tif"


def download_dem_tiles(bbox: dict, output_dir: Path, buffer_deg: float = 0.05) -> list[Path]:
    """Download all GLO-30 tiles covering bbox + buffer."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_ids = build_dem_tile_ids(
        west=bbox["west"] - buffer_deg,
        south=bbox["south"] - buffer_deg,
        east=bbox["east"] + buffer_deg,
        north=bbox["north"] + buffer_deg,
    )
    paths = []
    for tid in tile_ids:
        url = _tile_url(tid)
        dest = output_dir / f"cop_dem_glo30_{tid}.tif"
        if dest.exists():
            log.info("Already downloaded: %s", tid)
            paths.append(dest)
            continue
        log.info("Downloading DEM tile: %s", tid)
        try:
            urllib.request.urlretrieve(url, dest)
            paths.append(dest)
        except Exception as exc:
            log.warning("Failed to download %s: %s", tid, exc)
    return paths


def mosaic_and_reproject(
    tile_paths: list[Path],
    output_path: Path,
    target_crs: str,
    resolution_m: int = 30,
) -> Path:
    """Merge tiles, reproject to target CRS, save as COG."""
    if output_path.exists():
        log.info("DEM mosaic exists: %s", output_path.name)
        return output_path

    log.info("Mosaicking %d DEM tiles → %s", len(tile_paths), output_path.name)
    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = merge(datasets)
    for ds in datasets:
        ds.close()

    src_crs = datasets[0].crs
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, target_crs, mosaic.shape[2], mosaic.shape[1],
        resolution=resolution_m,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path, "w",
        driver="GTiff", height=dst_height, width=dst_width, count=1,
        dtype="float32", crs=target_crs,
        transform=dst_transform, nodata=-9999.0,
        compress="deflate", tiled=True,
    ) as dst:
        reproject(
            source=mosaic[0],
            destination=rasterio.band(dst, 1),
            src_transform=transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )

    log.info("DEM mosaic written: %s", output_path.name)
    return output_path
