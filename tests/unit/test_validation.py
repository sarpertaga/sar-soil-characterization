from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_bounds

from soilgeo.analysis.validation import (
    load_insitu_points,
    regression_metrics,
    sample_raster_at_points,
    validate_insitu,
)


def _make_raster(path: Path, value_fn) -> None:
    """A 100x100 EPSG:32636 raster whose pixel value = value_fn(row, col)."""
    rows = cols = 100
    data = np.fromfunction(np.vectorize(value_fn), (rows, cols), dtype=float).astype("float32")
    transform = from_bounds(500000, 4160000, 501000, 4161000, cols, rows)
    with rasterio.open(
        path, "w", driver="GTiff", height=rows, width=cols, count=1,
        dtype="float32", crs="EPSG:32636", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)


def _pixel_lonlat(path: Path, rc: list[tuple[int, int]]) -> list[tuple[float, float]]:
    """Lon/lat (EPSG:4326) of the centre of each (row, col) pixel."""
    with rasterio.open(path) as src:
        to4326 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        out = []
        for r, c in rc:
            x, y = src.xy(r, c)  # cell centre in raster CRS
            out.append(to4326.transform(x, y))
    return out


def _write_csv(path: Path, lonlat, obs, col="clay_pct_topsoil") -> None:
    lines = [f"lon,lat,{col}"]
    for (lon, lat), v in zip(lonlat, obs, strict=True):
        lines.append(f"{lon},{lat},{'' if v is None else v}")
    path.write_text("\n".join(lines) + "\n")


def test_sampling_recovers_pixel_values(tmp_path):
    ras = tmp_path / "pred.tif"
    _make_raster(ras, lambda r, c: r + c)  # value = row + col
    rc = [(10, 10), (50, 25), (90, 5)]
    lonlat = _pixel_lonlat(ras, rc)
    lon = np.array([p[0] for p in lonlat])
    lat = np.array([p[1] for p in lonlat])
    sampled = sample_raster_at_points(ras, lon, lat)
    expected = np.array([r + c for r, c in rc], dtype=float)
    assert np.allclose(sampled, expected, atol=1e-3)


def test_perfect_prediction_gives_r2_one(tmp_path):
    ras = tmp_path / "pred.tif"
    _make_raster(ras, lambda r, c: 2.0 * r + 1.0)
    rc = [(5, 5), (20, 60), (70, 30), (95, 95)]
    lonlat = _pixel_lonlat(ras, rc)
    obs = [2.0 * r + 1.0 for r, c in rc]  # observations == raster values
    csv = tmp_path / "insitu.csv"
    _write_csv(csv, lonlat, obs)
    res = validate_insitu(ras, csv, "clay_pct_topsoil", "clay")
    assert res["model"]["n"] == 4
    assert res["model"]["r2"] > 0.999
    assert res["model"]["rmse"] < 1e-3


def test_baseline_comparison_and_skill(tmp_path):
    pred = tmp_path / "pred.tif"
    base = tmp_path / "soilgrids.tif"
    _make_raster(pred, lambda r, c: float(r))           # good: matches obs
    _make_raster(base, lambda r, c: float(r) + 10.0)    # biased by +10
    rc = [(10, 10), (40, 40), (80, 20)]
    lonlat = _pixel_lonlat(pred, rc)
    obs = [float(r) for r, c in rc]
    csv = tmp_path / "insitu.csv"
    _write_csv(csv, lonlat, obs)
    res = validate_insitu(pred, csv, "clay_pct_topsoil", "clay", baseline_path=base)
    assert res["model"]["rmse"] < res["baseline_soilgrids"]["rmse"]
    assert res["model_beats_soilgrids"] is True
    assert res["rmse_improvement_vs_soilgrids"] > 0


def test_empty_value_rows_dropped(tmp_path):
    ras = tmp_path / "pred.tif"
    _make_raster(ras, lambda r, c: float(r + c))
    rc = [(10, 10), (20, 20)]
    lonlat = _pixel_lonlat(ras, rc)
    csv = tmp_path / "insitu.csv"
    _write_csv(csv, lonlat, [5.0, None])  # second row has no measured value
    lon, lat, obs = load_insitu_points(csv, "clay_pct_topsoil")
    assert len(obs) == 1


def test_regression_metrics_insufficient_points():
    res = regression_metrics(np.array([1.0]), np.array([1.0]))
    assert res["n"] == 1
    assert res["r2"] is None
