"""
Soil property regression: Random Forest with k-fold cross-validation.

Workflow:
    1. train_soil_model()   → fit RF, evaluate with k-fold CV
    2. predict_raster()     → apply model to full-resolution feature stack
"""
import json
from pathlib import Path

import numpy as np
import rasterio
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from soilgeo.utils.logging import get_logger

log = get_logger(__name__)

NODATA = -9999.0


def train_soil_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    target_name: str,
    n_estimators: int = 200,
    cv_folds: int = 5,
    random_state: int = 42,
) -> tuple[RandomForestRegressor, dict]:
    """
    Train a Random Forest regressor with k-fold cross-validation.

    Returns:
        model:    fitted RF on full training set (for prediction)
        metrics:  dict with cv RMSE / R² / MAE and feature importances
    """
    log.info("Training RF for '%s': n=%d features=%d", target_name, len(y), X.shape[1])

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_rmse, fold_r2, fold_mae = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=random_state,
        )
        rf.fit(X[train_idx], y[train_idx])
        y_pred = rf.predict(X[val_idx])

        rmse = float(np.sqrt(mean_squared_error(y[val_idx], y_pred)))
        r2   = float(r2_score(y[val_idx], y_pred))
        mae  = float(mean_absolute_error(y[val_idx], y_pred))
        fold_rmse.append(rmse)
        fold_r2.append(r2)
        fold_mae.append(mae)
        log.info("  fold %d/%d  RMSE=%.3f  R²=%.3f  MAE=%.3f", fold + 1, cv_folds, rmse, r2, mae)

    cv_metrics = {
        "target": target_name,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "cv_folds": cv_folds,
        "rmse_mean": float(np.mean(fold_rmse)),
        "rmse_std":  float(np.std(fold_rmse)),
        "r2_mean":   float(np.mean(fold_r2)),
        "r2_std":    float(np.std(fold_r2)),
        "mae_mean":  float(np.mean(fold_mae)),
        "mae_std":   float(np.std(fold_mae)),
    }
    log.info(
        "CV summary — RMSE=%.3f±%.3f  R²=%.3f±%.3f  MAE=%.3f±%.3f",
        cv_metrics["rmse_mean"], cv_metrics["rmse_std"],
        cv_metrics["r2_mean"],   cv_metrics["r2_std"],
        cv_metrics["mae_mean"],  cv_metrics["mae_std"],
    )

    # Fit final model on all training data
    model = RandomForestRegressor(
        n_estimators=n_estimators, n_jobs=-1, random_state=random_state
    )
    model.fit(X, y)

    cv_metrics["feature_importances"] = {
        name: float(imp)
        for name, imp in zip(feature_names, model.feature_importances_)
    }
    return model, cv_metrics


def predict_raster(
    model: RandomForestRegressor,
    X_full: np.ndarray,
    shape: tuple[int, int],
    profile: dict,
    output_path: Path,
) -> Path:
    """
    Apply a trained model to the full-resolution feature matrix and write GeoTIFF.

    Pixels where any feature is NODATA are set to NODATA in output.
    """
    if output_path.exists():
        log.info("Skipping prediction (exists): %s", output_path.name)
        return output_path

    valid_mask = np.all(X_full != NODATA, axis=1)
    pred = np.full(X_full.shape[0], NODATA, dtype=np.float32)
    if valid_mask.sum() > 0:
        pred[valid_mask] = model.predict(X_full[valid_mask]).astype(np.float32)

    pred_2d = pred.reshape(shape)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    p = profile.copy()
    p.update(count=1, dtype="float32", nodata=NODATA, compress="deflate")
    with rasterio.open(output_path, "w", **p) as dst:
        dst.write(pred_2d, 1)

    valid_vals = pred[valid_mask]
    if valid_vals.size > 0:
        log.info(
            "Prediction written: %s | range [%.2f, %.2f] mean=%.2f",
            output_path.name,
            float(valid_vals.min()), float(valid_vals.max()), float(valid_vals.mean()),
        )
    else:
        log.warning("Prediction written: %s | all pixels are nodata", output_path.name)
    return output_path


def save_metrics(metrics_list: list[dict], output_path: Path) -> Path:
    """Write a list of per-target CV metrics dicts to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics_list, indent=2))
    log.info("Metrics saved: %s", output_path.name)
    return output_path
