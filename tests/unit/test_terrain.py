import rasterio
from soilgeo.terrain.derivatives import compute_slope, compute_aspect, compute_roughness, compute_hillshade


def test_compute_slope_range(synthetic_dem_raster, tmp_path):
    out = tmp_path / "slope.tif"
    compute_slope(synthetic_dem_raster, out)
    assert out.exists()
    with rasterio.open(out) as src:
        data = src.read(1)
        valid = data[data != -9999.0]
        assert valid.min() >= 0.0
        assert valid.max() <= 90.0


def test_compute_aspect_range(synthetic_dem_raster, tmp_path):
    out = tmp_path / "aspect.tif"
    compute_aspect(synthetic_dem_raster, out)
    with rasterio.open(out) as src:
        data = src.read(1)
        valid = data[data != -9999.0]
        assert valid.min() >= 0.0
        assert valid.max() <= 360.0


def test_compute_roughness_non_negative(synthetic_dem_raster, tmp_path):
    out = tmp_path / "roughness.tif"
    compute_roughness(synthetic_dem_raster, out)
    with rasterio.open(out) as src:
        data = src.read(1)
        valid = data[data != -9999.0]
        assert (valid >= 0).all()


def test_compute_hillshade_range(synthetic_dem_raster, tmp_path):
    out = tmp_path / "hillshade.tif"
    compute_hillshade(synthetic_dem_raster, out)
    with rasterio.open(out) as src:
        data = src.read(1)
        valid = data[data != -9999.0]
        assert valid.min() >= 0
        assert valid.max() <= 255
