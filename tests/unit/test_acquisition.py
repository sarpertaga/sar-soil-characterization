from soilgeo.acquisition.sentinel_hub import build_sh_bbox, build_sh_config
from soilgeo.acquisition.dem import build_dem_tile_ids


def test_build_sh_bbox_correct_order():
    bbox = build_sh_bbox(west=32.20, south=37.55, east=33.20, north=38.20)
    assert bbox.min_x == 32.20
    assert bbox.min_y == 37.55
    assert bbox.max_x == 33.20
    assert bbox.max_y == 38.20


def test_build_sh_config_reads_env(monkeypatch):
    monkeypatch.setenv("SH_CLIENT_ID", "test_id")
    monkeypatch.setenv("SH_CLIENT_SECRET", "test_secret")
    cfg = build_sh_config()
    assert cfg.sh_client_id == "test_id"
    assert cfg.sh_client_secret == "test_secret"


def test_build_dem_tile_ids_konya():
    tile_ids = build_dem_tile_ids(west=32.20, south=37.55, east=33.20, north=38.20)
    assert "N37_E032" in tile_ids
    assert "N38_E032" in tile_ids
    assert len(tile_ids) >= 2


def test_build_dem_tile_ids_naming_format():
    tile_ids = build_dem_tile_ids(west=32.0, south=37.0, east=33.0, north=38.0)
    for tid in tile_ids:
        assert tid[0] in ("N", "S")
        assert "E" in tid or "W" in tid
