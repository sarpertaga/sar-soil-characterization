"""Fetch Sentinel-2 L2A bare-soil composite via Sentinel Hub CDSE."""
import os
from pathlib import Path

import numpy as np
import rasterio
from dotenv import load_dotenv

from soilgeo.acquisition.sentinel_hub import (
    CDSE_BASE, CDSE_TOKEN_URL, build_sh_bbox, build_sh_config, _save_tiff,
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


def fetch_s2_bare_soil(
    bbox_wgs84: dict,
    time_interval: tuple[str, str],
    output_path: Path,
    resolution_m: int = 10,
    ndvi_threshold: float = 0.4,
) -> Path:
    """
    Fetch Sentinel-2 L2A bare-soil composite for the given time window.

    Output: 7-band float32 GeoTIFF (reflectance [0,1]):
        band 1=B02  2=B03  3=B04  4=B08  5=B8A  6=B11  7=B12
    Pixels covered by cloud / water / snow / dense vegetation (NDVI > threshold)
    in ALL observations are set to NODATA.
    The ndvi_threshold argument is informational only — the filter is baked
    into the evalscript at 0.4; pass the same value for provenance tracking.
    """
    if output_path.exists():
        log.info("Skipping S2 fetch (exists): %s", output_path.name)
        return output_path

    from sentinelhub import MimeType, SentinelHubRequest, bbox_to_dimensions

    s2_col = _cdse_s2_collection()
    config = build_sh_config()
    sh_bbox = build_sh_bbox(**bbox_wgs84)
    size = bbox_to_dimensions(sh_bbox, resolution=resolution_m)

    log.info("S2 fetch [%s→%s] size=%s res=%dm", time_interval[0], time_interval[1], size, resolution_m)

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

    raw = request.get_data()[0]  # (H, W, 8): 7 bands + valid_mask
    valid_mask = raw[:, :, 7]
    bands = raw[:, :, :7].astype(np.float32)
    bands[valid_mask == 0] = NODATA

    _save_tiff(
        bands, output_path, bbox_wgs84,
        {
            "bands": ",".join(S2_BAND_NAMES),
            "units": "reflectance",
            "ndvi_threshold": str(ndvi_threshold),
            "time_start": time_interval[0],
            "time_end": time_interval[1],
        },
    )
    return output_path
