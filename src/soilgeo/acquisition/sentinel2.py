"""Fetch Sentinel-2 L2A bare-soil composite via Sentinel Hub CDSE."""
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds
from dotenv import load_dotenv

from soilgeo.acquisition.sentinel_hub import (
    CDSE_BASE, build_sh_bbox, build_sh_config,
)
from soilgeo.sar.evalscripts import S2_BARE_SOIL_COMPOSITE
from soilgeo.utils.logging import get_logger

load_dotenv()
log = get_logger(__name__)

NODATA = -9999.0

# Band order as output by S2_BARE_SOIL_COMPOSITE evalscript
S2_BAND_NAMES = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]
S2_BAND_IDX = {name: i for i, name in enumerate(S2_BAND_NAMES)}  # 0-based

_CDSE_S2 = None


def _cdse_s2_collection():
    global _CDSE_S2
    from sentinelhub import DataCollection
    if _CDSE_S2 is None:
        _CDSE_S2 = DataCollection.SENTINEL2_L2A.define_from(
            "CDSE_SENTINEL2_L2A", service_url=CDSE_BASE
        )
    return _CDSE_S2


def _split_bbox(bbox: dict, nx: int, ny: int) -> list[dict]:
    """Split a WGS84 bbox into nx×ny tiles."""
    w, s, e, n = bbox["west"], bbox["south"], bbox["east"], bbox["north"]
    dx = (e - w) / nx
    dy = (n - s) / ny
    tiles = []
    for row in range(ny):
        for col in range(nx):
            tiles.append({
                "west":  w + col * dx,
                "east":  w + (col + 1) * dx,
                "south": s + row * dy,
                "north": s + (row + 1) * dy,
            })
    return tiles


def _fetch_tile(
    tile_bbox: dict,
    time_interval: tuple[str, str],
    tile_path: Path,
    resolution_m: int,
) -> Path:
    """Fetch a single S2 tile. Skips if already exists."""
    if tile_path.exists():
        log.info("  tile exists: %s", tile_path.name)
        return tile_path

    from sentinelhub import MimeType, SentinelHubRequest, bbox_to_dimensions

    s2_col = _cdse_s2_collection()
    config = build_sh_config()
    sh_bbox = build_sh_bbox(**tile_bbox)
    size = bbox_to_dimensions(sh_bbox, resolution=resolution_m)

    log.info("  fetching tile %s size=%s", tile_path.name, size)

    request = SentinelHubRequest(
        evalscript=S2_BARE_SOIL_COMPOSITE,
        input_data=[SentinelHubRequest.input_data(
            data_collection=s2_col,
            time_interval=time_interval,
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=sh_bbox,
        size=size,
        config=config,
    )

    raw = request.get_data()[0]  # (H, W, 8)
    valid_mask = raw[:, :, 7]
    bands = raw[:, :, :7].astype(np.float32)
    bands[valid_mask == 0] = NODATA

    h, w = bands.shape[:2]
    transform = from_bounds(
        tile_bbox["west"], tile_bbox["south"],
        tile_bbox["east"], tile_bbox["north"],
        w, h,
    )
    tile_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        tile_path, "w",
        driver="GTiff", height=h, width=w, count=7,
        dtype="float32", crs="EPSG:4326",
        transform=transform, nodata=NODATA,
        compress="deflate",
    ) as dst:
        for i in range(7):
            dst.write(bands[:, :, i], i + 1)

    return tile_path


def fetch_s2_bare_soil(
    bbox_wgs84: dict,
    time_interval: tuple[str, str],
    output_path: Path,
    resolution_m: int = 10,
    ndvi_threshold: float = 0.4,
    tile_grid: tuple[int, int] = (3, 3),
) -> Path:
    """
    Fetch Sentinel-2 L2A bare-soil composite for the given time window.

    Large AOIs are automatically split into a tile_grid (default 3×3) to avoid
    API timeouts. Tiles are fetched sequentially and mosaicked into one GeoTIFF.

    Output: 7-band float32 GeoTIFF (reflectance [0,1]):
        band 1=B02  2=B03  3=B04  4=B08  5=B8A  6=B11  7=B12
    """
    if output_path.exists():
        log.info("Skipping S2 fetch (exists): %s", output_path.name)
        return output_path

    nx, ny = tile_grid
    tiles = _split_bbox(bbox_wgs84, nx, ny)
    tile_dir = output_path.parent / "_tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)

    log.info("S2 fetch [%s→%s] | %dx%d tile grid | res=%dm",
             time_interval[0], time_interval[1], nx, ny, resolution_m)

    tile_paths = []
    for i, tile_bbox in enumerate(tiles):
        tile_path = tile_dir / f"{output_path.stem}_tile_{i:02d}.tif"
        _fetch_tile(tile_bbox, time_interval, tile_path, resolution_m)
        tile_paths.append(tile_path)
        log.info("  tile %d/%d done", i + 1, nx * ny)

    # Mosaic all tiles
    log.info("Mosaicking %d tiles → %s", len(tile_paths), output_path.name)
    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = merge(datasets, nodata=NODATA)
    for ds in datasets:
        ds.close()

    profile = datasets[0].profile.copy()
    profile.update(
        height=mosaic.shape[1], width=mosaic.shape[2],
        transform=transform, compress="deflate",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mosaic)
        dst.update_tags(
            bands=",".join(S2_BAND_NAMES),
            units="reflectance",
            ndvi_threshold=str(ndvi_threshold),
            time_start=time_interval[0],
            time_end=time_interval[1],
        )

    log.info("S2 composite saved: %s (%.1f MB)",
             output_path.name, output_path.stat().st_size / 1e6)
    return output_path
