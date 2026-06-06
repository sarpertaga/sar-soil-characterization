from soilgeo.acquisition.soilgrids import (
    build_wcs_url, SOILGRIDS_PROPERTIES, VALID_DEPTHS,
)
import pytest


def test_build_wcs_url_contains_property():
    url = build_wcs_url("clay", "0-5cm", 32.2, 37.55, 33.2, 38.2)
    assert "clay" in url
    assert "clay_0-5cm_mean" in url


def test_build_wcs_url_contains_bbox():
    url = build_wcs_url("sand", "0-5cm", 32.2, 37.55, 33.2, 38.2)
    assert "32.2" in url
    assert "38.2" in url
    assert "SUBSET=X" in url
    assert "SUBSET=Y" in url


def test_build_wcs_url_wcs_version():
    url = build_wcs_url("soc", "0-5cm", 32.2, 37.55, 33.2, 38.2)
    assert "VERSION=2.0.1" in url
    assert "REQUEST=GetCoverage" in url
    assert "FORMAT=image/tiff" in url


def test_invalid_property_raises():
    with pytest.raises(ValueError, match="Unknown property"):
        build_wcs_url("nitrogen", "0-5cm", 32.2, 37.55, 33.2, 38.2)


def test_invalid_depth_raises():
    with pytest.raises(ValueError, match="Unknown depth"):
        build_wcs_url("clay", "100-200cm", 32.2, 37.55, 33.2, 38.2)


def test_all_default_properties_defined():
    for prop in ["clay", "sand", "soc"]:
        assert prop in SOILGRIDS_PROPERTIES


def test_soilgrids_properties_have_units():
    for prop, meta in SOILGRIDS_PROPERTIES.items():
        assert "unit" in meta
        assert "map" in meta
