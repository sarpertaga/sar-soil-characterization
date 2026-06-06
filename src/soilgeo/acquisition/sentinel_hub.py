"""Fetch Sentinel-1 σ⁰ VV+VH GeoTIFFs via Sentinel Hub Processing API."""
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from dotenv import load_dotenv

from soilgeo.sar.evalscripts import S1_VV_VH_MEDIAN_DB, S1_VV_VH_SINGLE_DB
from soilgeo.utils.logging import get_logger

load_dotenv()
log = get_logger(__name__)

NODATA = -9999.0


def build_sh_config():
    from sentinelhub import SHConfig
    cfg = SHConfig()
    cfg.sh_client_id = os.environ["SH_CLIENT_ID"]
    cfg.sh_client_secret = os.environ["SH_CLIENT_SECRET"]
    return cfg


def build_sh_bbox(west: float, south: float, east: float, north: float):
    from sentinelhub import BBox, CRS
    return BBox(bbox=[west, south, east, north], crs=CRS.WGS84)


def fetch_backscatter(
    bbox_wgs84: dict,
    time_interval: tuple[str, str],
    output_path: Path,
    resolution_m: int = 10,
    median_composite: bool = True,
) -> Path:
    """
    Fetch Sentinel-1 IW σ⁰ VV+VH for a time interval via Sentinel Hub.
    Saves 2-band float32 GeoTIFF: band1=VV_dB, band2=VH_dB.
    """
    if output_path.exists():
        log.info("Skipping fetch (exists): %s", output_path.name)
        return output_path

    from sentinelhub import (
        DataCollection, MimeType, MosaickingOrder,
        SentinelHubRequest, bbox_to_dimensions,
    )

    config = build_sh_config()
    sh_bbox = build_sh_bbox(**bbox_wgs84)
    size = bbox_to_dimensions(sh_bbox, resolution=resolution_m)
    evalscript = S1_VV_VH_MEDIAN_DB if median_composite else S1_VV_VH_SINGLE_DB

    log.info("Requesting Sentinel Hub: %s → %s | size=%s", time_interval[0], time_interval[1], size)

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL1_IW,
                time_interval=time_interval,
                mosaicking_order=MosaickingOrder.LEAST_CC,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=sh_bbox,
        size=size,
        config=config,
    )

    data = request.get_data()[0]  # (H, W, 3): VV, VH, dataMask
    vv = data[:, :, 0].astype(np.float32)
    vh = data[:, :, 1].astype(np.float32)
    mask = data[:, :, 2] == 0
    vv[mask] = NODATA
    vh[mask] = NODATA

    h, w = vv.shape
    transform = from_bounds(
        bbox_wgs84["west"], bbox_wgs84["south"],
        bbox_wgs84["east"], bbox_wgs84["north"],
        w, h,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path, "w",
        driver="GTiff", height=h, width=w, count=2,
        dtype="float32", crs="EPSG:4326",
        transform=transform, nodata=NODATA,
        compress="deflate",
    ) as dst:
        dst.write(vv, 1)
        dst.write(vh, 2)
        dst.update_tags(
            band_1="VV_sigma0_dB", band_2="VH_sigma0_dB",
            time_start=time_interval[0], time_end=time_interval[1],
        )

    log.info("Saved: %s (%dx%d px)", output_path.name, w, h)
    return output_path
