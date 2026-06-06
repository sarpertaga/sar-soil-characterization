import numpy as np
import rasterio
from soilgeo.indices.sar import compute_vv_vh_ratio, compute_moisture_index


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


def test_moisture_index_midpoint(tmp_path, synthetic_10m_raster):
    with rasterio.open(synthetic_10m_raster) as src:
        data = src.read(1)
        profile = src.profile

    dry = tmp_path / "dry.tif"
    wet = tmp_path / "wet.tif"
    scene = tmp_path / "scene.tif"
    for p, val in [(dry, -20.0), (wet, -10.0), (scene, -15.0)]:
        with rasterio.open(p, "w", **profile) as dst:
            dst.write(np.full_like(data, val), 1)

    mi_path = tmp_path / "mi.tif"
    compute_moisture_index(scene, dry, wet, mi_path)

    with rasterio.open(mi_path) as src:
        mi = src.read(1)
    valid = mi[mi != -9999.0]
    np.testing.assert_allclose(valid.mean(), 0.5, atol=0.05)


def test_moisture_index_clamped_0_1(tmp_path, synthetic_10m_raster):
    with rasterio.open(synthetic_10m_raster) as src:
        data = src.read(1)
        profile = src.profile

    dry = tmp_path / "dry.tif"
    wet = tmp_path / "wet.tif"
    scene = tmp_path / "scene_extreme.tif"
    for p, val in [(dry, -18.0), (wet, -8.0), (scene, -5.0)]:
        with rasterio.open(p, "w", **profile) as dst:
            dst.write(np.full_like(data, val), 1)

    mi_path = tmp_path / "mi_extreme.tif"
    compute_moisture_index(scene, dry, wet, mi_path)
    with rasterio.open(mi_path) as src:
        mi = src.read(1)
    valid = mi[mi != -9999.0]
    assert valid.max() <= 1.0
    assert valid.min() >= 0.0
