from soilgeo.acquisition.sentinel_hub import build_sh_bbox, build_sh_config


def test_build_sh_bbox_correct_order():
    bbox = build_sh_bbox(west=33.20, south=37.71, east=33.41, north=37.89)
    assert bbox.min_x == 33.20
    assert bbox.min_y == 37.71
    assert bbox.max_x == 33.41
    assert bbox.max_y == 37.89


def test_build_sh_config_reads_env(monkeypatch):
    monkeypatch.setenv("SH_CLIENT_ID", "test_id")
    monkeypatch.setenv("SH_CLIENT_SECRET", "test_secret")
    cfg = build_sh_config()
    assert cfg.sh_client_id == "test_id"
    assert cfg.sh_client_secret == "test_secret"


def test_build_sh_config_sets_cdse_endpoints(monkeypatch):
    monkeypatch.setenv("SH_CLIENT_ID", "test_id")
    monkeypatch.setenv("SH_CLIENT_SECRET", "test_secret")
    cfg = build_sh_config()
    assert "dataspace.copernicus.eu" in cfg.sh_base_url
    assert "dataspace.copernicus.eu" in cfg.sh_token_url
