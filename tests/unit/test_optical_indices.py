import numpy as np
import rasterio
from rasterio.transform import from_bounds

from soilgeo.indices.optical import (
    compute_bsi,
    compute_clay_index,
    compute_iron_oxide,
    compute_ndvi,
    compute_ndwi,
)

NODATA = -9999.0


def _make_s2_composite(tmp_path, band_values: dict[str, float]) -> object:
    """Create a synthetic 7-band S2 composite with uniform reflectance values."""
    band_order = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]
    path = tmp_path / "s2_test.tif"
    h, w = 20, 20
    transform = from_bounds(32.2, 37.55, 33.2, 38.2, w, h)
    with rasterio.open(
        path, "w",
        driver="GTiff", height=h, width=w, count=7,
        dtype="float32", crs="EPSG:4326",
        transform=transform, nodata=NODATA,
    ) as dst:
        for i, name in enumerate(band_order, start=1):
            val = band_values.get(name, 0.1)
            dst.write(np.full((h, w), val, dtype=np.float32), i)
    return path


def test_bsi_range(tmp_path):
    # Typical reflectance values for bare soil
    path = _make_s2_composite(tmp_path, {"B02": 0.08, "B04": 0.12, "B08": 0.15, "B11": 0.25})
    out = tmp_path / "bsi.tif"
    compute_bsi(path, out)
    with rasterio.open(out) as src:
        vals = src.read(1).flatten()
    valid = vals[vals != NODATA]
    assert valid.size > 0
    assert valid.min() >= -1.0
    assert valid.max() <= 1.0


def test_clay_index_positive_for_typical_soil(tmp_path):
    # B11 typically > B12 for clay-rich soils
    path = _make_s2_composite(tmp_path, {"B11": 0.25, "B12": 0.20})
    out = tmp_path / "clay_idx.tif"
    compute_clay_index(path, out)
    with rasterio.open(out) as src:
        vals = src.read(1).flatten()
    valid = vals[vals != NODATA]
    assert valid.mean() > 1.0   # B11/B12 > 1 when B11 > B12


def test_ndvi_range(tmp_path):
    path = _make_s2_composite(tmp_path, {"B04": 0.08, "B08": 0.35})
    out = tmp_path / "ndvi.tif"
    compute_ndvi(path, out)
    with rasterio.open(out) as src:
        vals = src.read(1).flatten()
    valid = vals[vals != NODATA]
    assert valid.min() >= -1.0
    assert valid.max() <= 1.0


def test_ndwi_inverted_for_dry_soil(tmp_path):
    # Dry soil: B08 >> B03 → NDWI < 0
    path = _make_s2_composite(tmp_path, {"B03": 0.07, "B08": 0.35})
    out = tmp_path / "ndwi.tif"
    compute_ndwi(path, out)
    with rasterio.open(out) as src:
        vals = src.read(1).flatten()
    valid = vals[vals != NODATA]
    assert valid.mean() < 0.0


def test_iron_oxide_greater_one_for_red_soil(tmp_path):
    # Red/iron-rich soil: B04 > B02
    path = _make_s2_composite(tmp_path, {"B02": 0.05, "B04": 0.15})
    out = tmp_path / "ior.tif"
    compute_iron_oxide(path, out)
    with rasterio.open(out) as src:
        vals = src.read(1).flatten()
    valid = vals[vals != NODATA]
    assert valid.mean() > 1.0


def test_indices_skip_if_exists(tmp_path):
    path = _make_s2_composite(tmp_path, {})
    out = tmp_path / "bsi.tif"
    out.write_bytes(b"dummy")  # pre-existing file
    original_mtime = out.stat().st_mtime
    compute_bsi(path, out)
    assert out.stat().st_mtime == original_mtime  # not overwritten
