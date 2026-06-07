"""
Gradient-boosting strong baseline (V3-F4).

Trains LightGBM (default) or XGBoost on the extended V3 feature matrix
(time-series statistics + climate + V2 features) using **spatial-block
GroupKFold** cross-validation, so that the reported metrics are not inflated
by spatial autocorrelation (risk R4). This is the bar the U-Net must clear to
justify its complexity.

Feature matrix layout matches V2: one row per sample pixel, columns =
``feature_names``. The ``groups`` array carries each sample's spatial-block id
(from :mod:`soilgeo.models.tiling`), and GroupKFold guarantees that all samples
from one block stay within a single fold.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from soilgeo.utils.logging import get_logger

log = get_logger(__name__)

NODATA = -9999.0


def _make_regressor(backend: str, n_estimators: int, learning_rate: float, random_state: int):
    if backend == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    if backend == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"unknown backend: {backend!r} (expected 'lightgbm' or 'xgboost')")


def train_gbm_spatial_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    target_name: str,
    backend: str = "lightgbm",
    n_splits: int = 5,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> tuple[object, dict]:
    """
    Train a gradient-boosting regressor with spatial-block GroupKFold CV.

    Returns:
        model:   regressor refit on the full dataset (for prediction)
        metrics: dict with per-fold and mean R²/RMSE/MAE + feature importances
    """
    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_groups)
    log.info(
        "Training %s for '%s': n=%d features=%d groups=%d splits=%d",
        backend, target_name, len(y), X.shape[1], n_groups, n_splits,
    )

    gkf = GroupKFold(n_splits=n_splits)
    fold_rmse, fold_r2, fold_mae = [], [], []

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        reg = _make_regressor(backend, n_estimators, learning_rate, random_state)
        reg.fit(X[tr], y[tr])
        pred = reg.predict(X[va])
        rmse = float(np.sqrt(mean_squared_error(y[va], pred)))
        r2 = float(r2_score(y[va], pred))
        mae = float(mean_absolute_error(y[va], pred))
        fold_rmse.append(rmse)
        fold_r2.append(r2)
        fold_mae.append(mae)
        log.info("  fold %d/%d  RMSE=%.3f  R²=%.3f  MAE=%.3f", fold + 1, n_splits, rmse, r2, mae)

    model = _make_regressor(backend, n_estimators, learning_rate, random_state)
    model.fit(X, y)

    metrics = {
        "target": target_name,
        "backend": backend,
        "cv_strategy": "spatial_block_groupkfold",
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_splits": n_splits,
        "rmse_mean": float(np.mean(fold_rmse)), "rmse_std": float(np.std(fold_rmse)),
        "r2_mean": float(np.mean(fold_r2)),     "r2_std": float(np.std(fold_r2)),
        "mae_mean": float(np.mean(fold_mae)),   "mae_std": float(np.std(fold_mae)),
        "feature_importances": {
            name: float(imp)
            for name, imp in zip(feature_names, model.feature_importances_)
        },
    }
    log.info(
        "Spatial-CV summary — RMSE=%.3f±%.3f  R²=%.3f±%.3f  MAE=%.3f±%.3f",
        metrics["rmse_mean"], metrics["rmse_std"],
        metrics["r2_mean"], metrics["r2_std"],
        metrics["mae_mean"], metrics["mae_std"],
    )
    return model, metrics


def predict_gbm(model: object, X_full: np.ndarray) -> np.ndarray:
    """
    Apply a trained GBM to a feature matrix. Rows containing any NODATA feature
    are returned as NODATA (consistent with V2 ``predict_raster``).
    """
    valid = np.all(X_full != NODATA, axis=1)
    out = np.full(X_full.shape[0], NODATA, dtype=np.float32)
    if valid.sum() > 0:
        out[valid] = model.predict(X_full[valid]).astype(np.float32)
    return out


def save_metrics(metrics_list: list[dict], output_path: Path) -> Path:
    """Write a list of per-target GBM metrics dicts to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics_list, indent=2))
    log.info("GBM metrics saved: %s", output_path.name)
    return output_path
