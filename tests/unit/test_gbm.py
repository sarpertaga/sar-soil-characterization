import numpy as np
import pytest

# GBM backends live in the soilgeo-dl env; skip cleanly under the core env.
pytest.importorskip("lightgbm")

from soilgeo.models.gbm import (  # noqa: E402
    predict_gbm,
    train_gbm_quantile_cv,
    train_gbm_spatial_cv,
)


def _learnable_dataset(n=600, seed=0):
    """Feature matrix with a clear linear signal + spatial block groups."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, 5)).astype(np.float32)
    y = (3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * X[:, 2]).astype(np.float32)
    y += rng.normal(0, 0.1, size=n).astype(np.float32)
    # 12 spatial blocks, contiguous chunks of samples share a block
    groups = np.repeat(np.arange(12), n // 12)
    return X, y, groups, ["f0", "f1", "f2", "f3", "f4"]


def test_returns_model_and_metrics():
    X, y, groups, names = _learnable_dataset()
    model, metrics = train_gbm_spatial_cv(
        X, y, groups, names, target_name="clay", n_splits=4, n_estimators=50
    )
    assert "r2_mean" in metrics and "rmse_mean" in metrics and "mae_mean" in metrics
    assert metrics["target"] == "clay"
    assert metrics["cv_strategy"] == "spatial_block_groupkfold"


def test_learns_signal():
    X, y, groups, names = _learnable_dataset()
    _model, metrics = train_gbm_spatial_cv(
        X, y, groups, names, target_name="clay", n_splits=4, n_estimators=100
    )
    # strong learnable signal -> spatial-CV R² should be clearly positive
    assert metrics["r2_mean"] > 0.7


def test_feature_importances_present_and_named():
    X, y, groups, names = _learnable_dataset()
    _model, metrics = train_gbm_spatial_cv(
        X, y, groups, names, target_name="clay", n_splits=4, n_estimators=50
    )
    imp = metrics["feature_importances"]
    assert set(imp.keys()) == set(names)
    # f0 (largest coefficient) should carry meaningful importance
    assert imp["f0"] > 0


def test_quantile_cv_returns_models_and_calibration():
    X, y, groups, _names = _learnable_dataset(n=1200)
    models, metrics = train_gbm_quantile_cv(
        X, y, groups, target_name="clay", n_splits=4, n_estimators=80
    )
    assert set(models.keys()) == {0.05, 0.5, 0.95}
    assert metrics["nominal_coverage"] == 0.9
    # empirical coverage should be in a sane range and interval width positive
    assert 0.5 < metrics["empirical_coverage"] <= 1.0
    assert metrics["mean_interval_width"] > 0


def test_quantile_intervals_are_ordered():
    X, y, groups, _names = _learnable_dataset(n=900)
    models, _m = train_gbm_quantile_cv(
        X, y, groups, target_name="clay", n_splits=3, n_estimators=80
    )
    q05 = models[0.05].predict(X)
    q95 = models[0.95].predict(X)
    # upper quantile should sit above the lower on average
    assert np.mean(q95 - q05) > 0


def test_predict_shape_and_nodata():
    X, y, groups, names = _learnable_dataset()
    model, _metrics = train_gbm_spatial_cv(
        X, y, groups, names, target_name="clay", n_splits=4, n_estimators=50
    )
    X_full = X.copy()
    X_full[0, :] = -9999.0          # a nodata row
    pred = predict_gbm(model, X_full)
    assert pred.shape == (X_full.shape[0],)
    assert pred[0] == -9999.0       # nodata propagated
    assert pred[1] != -9999.0
