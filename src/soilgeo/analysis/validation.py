"""Independent in-situ validation against real soil observations (WoSIS).

The V1–V3 models are *trained* on SoilGrids 250 m labels (a model product), so
their cross-validated R² measures agreement with another model, not with ground
truth. This module closes that gap: it samples a predicted raster at real,
held-out WoSIS point locations and reports prediction error against measured
clay/sand/SOC — and, crucially, computes the same error for SoilGrids itself at
those points, so the model can be judged *relative to the label source*.

These points are NEVER used in training; they are an external test set only.
"""

import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from soilgeo.utils.geo import NODATA
from soilgeo.utils.logging import get_logger

log = get_logger(__name__)


def load_insitu_points(csv_path: Path, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read lon, lat and one measured value column from a WoSIS validation CSV.

    Returns ``(lon, lat, obs)`` float arrays, dropping rows with an empty value.
    """
    lon, lat, obs = [], [], []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        if value_col not in reader.fieldnames:
            raise KeyError(f"{value_col!r} not in {csv_path.name}: {reader.fieldnames}")
        for row in reader:
            v = row.get(value_col, "").strip()
            if v == "":
                continue
            lon.append(float(row["lon"]))
            lat.append(float(row["lat"]))
            obs.append(float(v))
    log.info("Loaded %d in-situ points (%s) from %s", len(obs), value_col, csv_path.name)
    return np.array(lon), np.array(lat), np.array(obs)


def sample_raster_at_points(
    raster_path: Path, lon: np.ndarray, lat: np.ndarray, band: int = 1
) -> np.ndarray:
    """Sample a single raster band at lon/lat points (EPSG:4326).

    Points are reprojected to the raster CRS. Samples that fall outside the grid
    or on a nodata cell are returned as NaN.
    """
    with rasterio.open(raster_path) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = transformer.transform(lon, lat)
        nodata = src.nodata if src.nodata is not None else NODATA
        out = np.full(len(lon), np.nan, dtype="float64")
        for i, val in enumerate(src.sample(zip(xs, ys, strict=True), indexes=band)):
            v = float(val[0])
            if np.isfinite(v) and v != nodata:
                out[i] = v
    return out


def regression_metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    """R², RMSE, MAE, bias (pred − obs) over paired finite values."""
    mask = np.isfinite(obs) & np.isfinite(pred)
    n = int(mask.sum())
    if n < 2:
        return {"n": n, "r2": None, "rmse": None, "mae": None, "bias": None}
    o, p = obs[mask], pred[mask]
    return {
        "n": n,
        "r2": float(r2_score(o, p)),
        "rmse": float(np.sqrt(mean_squared_error(o, p))),
        "mae": float(mean_absolute_error(o, p)),
        "bias": float(np.mean(p - o)),
    }


def validate_insitu(
    pred_path: Path,
    insitu_csv: Path,
    value_col: str,
    target: str,
    baseline_path: Path | None = None,
    pred_scale: float = 1.0,
    baseline_scale: float = 1.0,
) -> dict:
    """Validate one predicted raster against in-situ observations.

    Args:
        pred_path: model prediction raster (e.g. ``clay_10m_goller_pilot.tif``).
        insitu_csv: WoSIS validation CSV (lon, lat, measured columns).
        value_col: CSV column with the measured value (e.g. ``clay_pct_topsoil``).
        target: short target name for reporting (``clay`` / ``sand`` / ``soc``).
        baseline_path: optional SoilGrids raster of the same property, sampled at
            the same points to report the label source's own error.
        pred_scale / baseline_scale: multipliers to bring rasters into the same
            unit as the observations (e.g. SoilGrids clay is g/kg = %×10).

    Returns a dict with model metrics and, if given, baseline metrics + the
    skill of the model relative to the baseline.
    """
    lon, lat, obs = load_insitu_points(insitu_csv, value_col)
    pred = sample_raster_at_points(pred_path, lon, lat) * pred_scale
    result = {
        "target": target,
        "insitu_source": insitu_csv.name,
        "value_col": value_col,
        "n_points_total": int(len(obs)),
        "model": regression_metrics(obs, pred),
    }

    if baseline_path is not None:
        base = sample_raster_at_points(baseline_path, lon, lat) * baseline_scale
        result["baseline_soilgrids"] = regression_metrics(obs, base)
        m, b = result["model"]["rmse"], result["baseline_soilgrids"]["rmse"]
        if m is not None and b is not None:
            result["rmse_improvement_vs_soilgrids"] = float(b - m)
            result["model_beats_soilgrids"] = bool(m < b)

    log.info(
        "in-situ %s: model R²=%s RMSE=%s (n=%s)",
        target, result["model"]["r2"], result["model"]["rmse"], result["model"]["n"],
    )
    return result


def save_validation(results: list[dict], output_path: Path) -> Path:
    """Write a list of per-target in-situ validation dicts to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    log.info("In-situ validation saved: %s", output_path.name)
    return output_path
