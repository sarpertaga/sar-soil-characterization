from soilgeo.sar.evalscripts import S2_BARE_SOIL_COMPOSITE
from soilgeo.acquisition.sentinel2 import S2_BAND_NAMES, S2_BAND_IDX


def test_evalscript_excludes_cloud_scl_values():
    assert "EXCLUDE_SCL" in S2_BARE_SOIL_COMPOSITE
    for scl in [8, 9, 10]:   # high/medium cloud, thin cirrus
        assert str(scl) in S2_BARE_SOIL_COMPOSITE


def test_evalscript_filters_ndvi():
    assert "ndvi" in S2_BARE_SOIL_COMPOSITE.lower()
    assert "0.4" in S2_BARE_SOIL_COMPOSITE


def test_evalscript_outputs_7_bands_plus_mask():
    # output declaration says bands: 8
    assert "bands: 8" in S2_BARE_SOIL_COMPOSITE


def test_band_names_order():
    assert S2_BAND_NAMES == ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]


def test_band_idx_mapping():
    assert S2_BAND_IDX["B02"] == 0
    assert S2_BAND_IDX["B11"] == 5
    assert S2_BAND_IDX["B12"] == 6
    assert len(S2_BAND_IDX) == 7
