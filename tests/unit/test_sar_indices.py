import numpy as np
import rasterio
from soilgeo.indices.sar import compute_vv_vh_ratio, compute_nddi


def test_vv_vh_ratio_is_difference(tmp_path, synthetic_10m_raster):
    vv_path = synthetic_10m_raster
    vh_path = tmp_path / "vh.tif"
    with rasterio.open(vv_path) as src:
        vv_data = src.read(1)
        profile = src.profile
    with rasterio.open(vh_path, "w", **profile) as dst:
        dst.write(vv_data - 5.0, 1)

    ratio_path = tmp_path / "ratio.tif"
    compute_vv_vh_ratio(vv_path, vh_path, ratio_path)

    with rasterio.open(ratio_path) as src:
        ratio = src.read(1)
    valid = ratio[ratio != -9999.0]
    np.testing.assert_allclose(valid, 5.0, atol=1e-3)


def test_nddi_range_minus1_to_1(tmp_path, synthetic_10m_raster):
    with rasterio.open(synthetic_10m_raster) as src:
        data = src.read(1)
        profile = src.profile

    wet = tmp_path / "wet.tif"
    dry = tmp_path / "dry.tif"
    # wet = -10 dB, dry = -15 dB → wet > dry in linear → NDDI positive
    for p, val in [(wet, -10.0), (dry, -15.0)]:
        with rasterio.open(p, "w", **profile) as dst:
            dst.write(np.full_like(data, val), 1)

    out = tmp_path / "nddi.tif"
    compute_nddi(wet, dry, out)

    with rasterio.open(out) as src:
        nddi = src.read(1)
    valid = nddi[nddi != -9999.0]
    assert valid.min() >= -1.0
    assert valid.max() <= 1.0


def test_nddi_positive_when_wet_greater(tmp_path, synthetic_10m_raster):
    with rasterio.open(synthetic_10m_raster) as src:
        data = src.read(1)
        profile = src.profile

    wet = tmp_path / "wet.tif"
    dry = tmp_path / "dry.tif"
    for p, val in [(wet, -10.0), (dry, -15.0)]:
        with rasterio.open(p, "w", **profile) as dst:
            dst.write(np.full_like(data, val), 1)

    out = tmp_path / "nddi.tif"
    compute_nddi(wet, dry, out)
    with rasterio.open(out) as src:
        nddi = src.read(1)
    assert nddi[nddi != -9999.0].mean() > 0


def test_nddi_zero_when_equal(tmp_path, synthetic_10m_raster):
    with rasterio.open(synthetic_10m_raster) as src:
        data = src.read(1)
        profile = src.profile

    same = tmp_path / "same.tif"
    with rasterio.open(same, "w", **profile) as dst:
        dst.write(np.full_like(data, -12.0), 1)

    out = tmp_path / "nddi_zero.tif"
    compute_nddi(same, same, out)
    with rasterio.open(out) as src:
        nddi = src.read(1)
    valid = nddi[nddi != -9999.0]
    np.testing.assert_allclose(valid, 0.0, atol=1e-5)
