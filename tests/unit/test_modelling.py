from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from soilgeo.modelling.regression import predict_raster, save_metrics, train_soil_model

# ─── helpers ─────────────────────────────────────────────────────────────────

def _make_raster(tmp_path: Path, name: str, value: float, shape=(30, 30)) -> Path:
    path = tmp_path / f"{name}.tif"
    h, w = shape
    transform = from_bounds(32.2, 37.55, 33.2, 38.2, w, h)
    data = np.full((h, w), value, dtype=np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return path


# ─── regression tests ────────────────────────────────────────────────────────

def _synthetic_Xy(n=200, n_feat=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, n_feat)).astype(np.float32)
    # y = weighted sum + noise (learnable signal)
    weights = rng.uniform(1, 3, n_feat)
    y = (X @ weights + rng.normal(0, 0.5, n)).astype(np.float32)
    return X, y


def test_train_soil_model_returns_model_and_metrics():
    X, y = _synthetic_Xy()
    feat_names = [f"f{i}" for i in range(X.shape[1])]
    model, metrics = train_soil_model(X, y, feat_names, "clay", n_estimators=10, cv_folds=3)
    assert hasattr(model, "predict")
    assert "rmse_mean" in metrics
    assert "r2_mean" in metrics
    assert "mae_mean" in metrics


def test_train_soil_model_r2_positive_on_learnable_signal():
    X, y = _synthetic_Xy(n=500)
    feat_names = [f"f{i}" for i in range(X.shape[1])]
    _, metrics = train_soil_model(X, y, feat_names, "clay", n_estimators=50, cv_folds=3)
    assert metrics["r2_mean"] > 0.5


def test_train_soil_model_feature_importances_sum_to_one():
    X, y = _synthetic_Xy()
    feat_names = [f"f{i}" for i in range(X.shape[1])]
    _, metrics = train_soil_model(X, y, feat_names, "clay", n_estimators=10, cv_folds=2)
    imps = list(metrics["feature_importances"].values())
    assert abs(sum(imps) - 1.0) < 1e-5


def test_predict_raster_correct_shape(tmp_path):
    X, y = _synthetic_Xy(n=300)
    feat_names = [f"f{i}" for i in range(X.shape[1])]
    model, _ = train_soil_model(X, y, feat_names, "clay", n_estimators=10, cv_folds=2)

    shape = (10, 10)
    X_full = np.random.default_rng(1).uniform(0, 1, (100, X.shape[1])).astype(np.float32)
    profile = {
        "driver": "GTiff", "dtype": "float32", "crs": "EPSG:4326",
        "transform": from_bounds(32.2, 37.55, 33.2, 38.2, 10, 10),
        "height": 10, "width": 10, "count": 1, "nodata": -9999.0,
    }
    out = tmp_path / "pred.tif"
    predict_raster(model, X_full, shape, profile, out)

    assert out.exists()
    with rasterio.open(out) as src:
        data = src.read(1)
        assert data.shape == shape
        assert (data != -9999.0).any()


def test_predict_raster_nodata_for_nodata_pixels(tmp_path):
    X, y = _synthetic_Xy(n=200)
    feat_names = [f"f{i}" for i in range(X.shape[1])]
    model, _ = train_soil_model(X, y, feat_names, "clay", n_estimators=10, cv_folds=2)

    X_full = np.full((4, X.shape[1]), -9999.0, dtype=np.float32)
    profile = {
        "driver": "GTiff", "dtype": "float32", "crs": "EPSG:4326",
        "transform": from_bounds(32.2, 37.55, 33.2, 38.2, 2, 2),
        "height": 2, "width": 2, "count": 1, "nodata": -9999.0,
    }
    out = tmp_path / "pred_nodata.tif"
    predict_raster(model, X_full, (2, 2), profile, out)
    with rasterio.open(out) as src:
        data = src.read(1).flatten()
    assert (data == -9999.0).all()


def test_save_metrics(tmp_path):
    import json
    metrics = [{"target": "clay", "rmse_mean": 5.2, "r2_mean": 0.7}]
    out = tmp_path / "metrics.json"
    save_metrics(metrics, out)
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded[0]["target"] == "clay"
    assert loaded[0]["r2_mean"] == pytest.approx(0.7)
