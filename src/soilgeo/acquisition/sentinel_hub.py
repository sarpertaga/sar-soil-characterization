"""Fetch all raster data via Sentinel Hub Processing API (Copernicus Data Space)."""
import os
from pathlib import Path

import numpy as np
import rasterio
from dotenv import load_dotenv
from rasterio.transform import from_bounds

from soilgeo.sar.evalscripts import DEM_ELEVATION, S1_VV_VH_MEDIAN_DB, S1_VV_VH_SINGLE_DB
from soilgeo.utils.logging import get_logger

load_dotenv()
log = get_logger(__name__)

NODATA = -9999.0
CDSE_BASE = "https://sh.dataspace.copernicus.eu"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)

# Module-level collection cache — defined only once per process
_CDSE_S1 = None
_CDSE_DEM = None


def _cdse_collections():
    """Return (S1_collection, DEM_collection) pointing at CDSE endpoints."""
    global _CDSE_S1, _CDSE_DEM
    from sentinelhub import DataCollection
    if _CDSE_S1 is None:
        _CDSE_S1 = DataCollection.SENTINEL1_IW.define_from(
            "CDSE_SENTINEL1_IW", service_url=CDSE_BASE
        )
    if _CDSE_DEM is None:
        _CDSE_DEM = DataCollection.DEM.define_from(
            "CDSE_DEM_GLO30", service_url=CDSE_BASE
        )
    return _CDSE_S1, _CDSE_DEM


def build_sh_config():
    from sentinelhub import SHConfig
    cfg = SHConfig()
    cfg.sh_client_id = os.environ["SH_CLIENT_ID"]
    cfg.sh_client_secret = os.environ["SH_CLIENT_SECRET"]
    cfg.sh_base_url = CDSE_BASE
    cfg.sh_token_url = CDSE_TOKEN_URL
    return cfg


def build_sh_bbox(west: float, south: float, east: float, north: float):
    from sentinelhub import CRS, BBox
    return BBox(bbox=[west, south, east, north], crs=CRS.WGS84)


def _save_tiff(data: np.ndarray, output_path: Path, bbox_wgs84: dict, band_tags: dict):
    if data.ndim == 3:
        h, w, n_bands = data.shape
    else:
        h, w = data.shape
        n_bands = 1
        data = data[:, :, np.newaxis]

    transform = from_bounds(
        bbox_wgs84["west"], bbox_wgs84["south"],
        bbox_wgs84["east"], bbox_wgs84["north"],
        w, h,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path, "w",
        driver="GTiff", height=h, width=w, count=n_bands,
        dtype="float32", crs="EPSG:4326",
        transform=transform, nodata=NODATA,
        compress="deflate",
    ) as dst:
        for i in range(n_bands):
            dst.write(data[:, :, i].astype(np.float32), i + 1)
        dst.update_tags(**band_tags)
    log.info("Saved: %s (%dx%d, %d band(s))", output_path.name, w, h, n_bands)


def fetch_backscatter(
    bbox_wgs84: dict,
    time_interval: tuple[str, str],
    output_path: Path,
    resolution_m: int = 10,
    median_composite: bool = True,
    orbit_direction: str | None = None,
) -> Path:
    """
    Sentinel-1 IW σ⁰ median composite via CDSE Sentinel Hub.
    Output: 2-band float32 GeoTIFF — band1=VV_dB, band2=VH_dB.

    ``orbit_direction`` ("ASCENDING"/"DESCENDING") restricts acquisitions to a
    single pass direction so incidence-angle geometry stays consistent across
    dates (DR-R2); None mixes both passes.
    """
    if output_path.exists():
        log.info("Skipping (exists): %s", output_path.name)
        return output_path

    from sentinelhub import MimeType, SentinelHubRequest, bbox_to_dimensions

    s1_col, _ = _cdse_collections()
    config = build_sh_config()
    sh_bbox = build_sh_bbox(**bbox_wgs84)
    size = bbox_to_dimensions(sh_bbox, resolution=resolution_m)
    evalscript = S1_VV_VH_MEDIAN_DB if median_composite else S1_VV_VH_SINGLE_DB

    other_args = None
    if orbit_direction is not None:
        direction = orbit_direction.upper()
        if direction not in ("ASCENDING", "DESCENDING"):
            raise ValueError(f"invalid orbit_direction: {orbit_direction!r}")
        other_args = {"dataFilter": {"orbitDirection": direction}}

    log.info("S1 fetch [%s→%s] size=%s orbit=%s",
             time_interval[0], time_interval[1], size, orbit_direction or "any")

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(
            data_collection=s1_col,
            time_interval=time_interval,
            other_args=other_args,
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=sh_bbox,
        size=size,
        config=config,
    )

    data = request.get_data()[0]  # (H, W, 3): VV, VH, dataMask
    vv = data[:, :, 0].astype(np.float32)
    vh = data[:, :, 1].astype(np.float32)
    mask = data[:, :, 2] == 0
    vv[mask | ~np.isfinite(vv)] = NODATA
    vh[mask | ~np.isfinite(vh)] = NODATA

    _save_tiff(
        np.stack([vv, vh], axis=2), output_path, bbox_wgs84,
        {"band_1": "VV_sigma0_dB", "band_2": "VH_sigma0_dB",
         "time_start": time_interval[0], "time_end": time_interval[1]},
    )
    return output_path


def fetch_dem(
    bbox_wgs84: dict,
    output_path: Path,
    resolution_m: int = 30,
) -> Path:
    """
    Copernicus DEM elevation via CDSE Sentinel Hub.
    Output: single-band float32 GeoTIFF — elevation in metres.
    """
    if output_path.exists():
        log.info("Skipping (exists): %s", output_path.name)
        return output_path

    from sentinelhub import MimeType, SentinelHubRequest, bbox_to_dimensions

    _, dem_col = _cdse_collections()
    config = build_sh_config()
    sh_bbox = build_sh_bbox(**bbox_wgs84)
    size = bbox_to_dimensions(sh_bbox, resolution=resolution_m)

    log.info("DEM fetch: size=%s resolution=%dm", size, resolution_m)

    request = SentinelHubRequest(
        evalscript=DEM_ELEVATION,
        input_data=[SentinelHubRequest.input_data(data_collection=dem_col)],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=sh_bbox,
        size=size,
        config=config,
    )

    data = request.get_data()[0]
    elev = data[:, :, 0] if data.ndim == 3 else data

    _save_tiff(
        elev.astype(np.float32)[:, :, np.newaxis], output_path, bbox_wgs84,
        {"band_1": "elevation_m", "source": "Copernicus_DEM_GLO30"},
    )
    return output_path
