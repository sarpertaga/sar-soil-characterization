import json
import logging
from pathlib import Path

import numpy as np
import rasterio
import pytest

from soilgeo.utils.logging import get_logger
from soilgeo.utils.config import load_aoi_config, load_pipeline_config, AoiConfig, PipelineConfig
from soilgeo.utils.geo import write_cog, bbox_to_utm, read_band


# ── Logging ──────────────────────────────────────────────────────────────────

def test_get_logger_returns_logger():
    logger = get_logger("soilgeo.test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "soilgeo.test"


def test_get_logger_same_name_returns_same_instance():
    a = get_logger("soilgeo.x")
    b = get_logger("soilgeo.x")
    assert a is b


# ── Config ────────────────────────────────────────────────────────────────────

FIXTURE_AOI = Path("config/aoi/konya.yml")
FIXTURE_PIPELINE = Path("config/pipelines/v1.yml")


def test_load_aoi_config_returns_dataclass():
    cfg = load_aoi_config(FIXTURE_AOI)
    assert isinstance(cfg, AoiConfig)
    assert cfg.name == "konya"
    assert cfg.crs == "EPSG:32636"
    assert cfg.resolution_m == 10


def test_aoi_config_bbox_valid():
    cfg = load_aoi_config(FIXTURE_AOI)
    assert cfg.bbox["west"] < cfg.bbox["east"]
    assert cfg.bbox["south"] < cfg.bbox["north"]


def test_load_pipeline_config():
    cfg = load_pipeline_config(FIXTURE_PIPELINE)
    assert isinstance(cfg, PipelineConfig)
    assert cfg.version == "v1"


# ── Geo utils ─────────────────────────────────────────────────────────────────

def test_write_cog_creates_valid_file(tmp_path, synthetic_10m_raster):
    out = tmp_path / "out.tif"
    with rasterio.open(synthetic_10m_raster) as src:
        data = src.read(1)
        profile = src.profile
    write_cog(out, data, profile)
    assert out.exists()
    with rasterio.open(out) as dst:
        assert dst.nodata == -9999.0
        assert dst.profile["compress"] == "deflate"


def test_write_cog_embeds_provenance(tmp_path, synthetic_10m_raster):
    out = tmp_path / "out.tif"
    with rasterio.open(synthetic_10m_raster) as src:
        data = src.read(1)
        profile = src.profile
    write_cog(out, data, profile, provenance={"source": "test", "version": "0.1"})
    sidecar = out.with_suffix(".json")
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["source"] == "test"


def test_bbox_to_utm_returns_valid_bounds():
    bbox_utm = bbox_to_utm(west=32.20, south=37.55, east=33.20, north=38.20, crs="EPSG:32636")
    assert bbox_utm["minx"] < bbox_utm["maxx"]
    assert bbox_utm["miny"] < bbox_utm["maxy"]


def test_read_band_returns_float32(synthetic_10m_raster):
    arr = read_band(synthetic_10m_raster, band=1)
    assert arr.dtype == np.float32
    assert arr.ndim == 2
