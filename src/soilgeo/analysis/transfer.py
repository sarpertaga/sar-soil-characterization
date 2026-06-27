"""Cross-region (out-of-region) transferability evaluation.

Spatial-block CV measures how well a model generalises to *unseen locations
within the same AOI*. It does not answer the harder question a soil-mapping
product actually faces: does a model trained in region A work in region B?

This module trains a GBM on one region's full feature matrix and evaluates it on
another region's matrix, after aligning both to their common feature set. The
resulting out-of-region R²/RMSE is typically far below the in-region spatial-CV
score — quantifying, honestly, how region-specific the learned signal is.
"""

import numpy as np

from soilgeo.models.gbm import _make_regressor
from soilgeo.utils.logging import get_logger

log = get_logger(__name__)


def common_features(names_a: list[str], names_b: list[str]) -> list[str]:
    """Sorted intersection of two feature-name lists."""
    return sorted(set(map(str, names_a)) & set(map(str, names_b)))


def align_features(X: np.ndarray, names: list[str], target: list[str]) -> np.ndarray:
    """Select/reorder columns of ``X`` so they match ``target`` order.

    Raises if any target feature is missing from ``names``.
    """
    names = [str(n) for n in names]
    idx = []
    for t in target:
        if t not in names:
            raise KeyError(f"feature {t!r} not in matrix ({names})")
        idx.append(names.index(t))
    return X[:, idx]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Out-of-region error metrics.

    Reports raw R² (penalises both miscalibration and lost correlation) *and*
    rank/linear correlation plus a bias-corrected R² (signal retained after
    removing the cross-region mean/scale offset), so a model that ranks pixels
    correctly but is miscalibrated is not confused with one that learned nothing.
    """
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    pear = float(pearsonr(y_true, y_pred)[0])
    spear = float(spearmanr(y_true, y_pred).correlation)
    # bias-correct: linearly rescale prediction to the test region (a*pred+b)
    a, b = np.polyfit(y_pred, y_true, 1)
    r2_bc = float(r2_score(y_true, a * y_pred + b))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "r2_bias_corrected": r2_bc,
        "pearson_r": pear,
        "spearman_rho": spear,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "n": int(len(y_true)),
    }


def cross_region_transfer(
    X_train: np.ndarray, y_train: np.ndarray, names_train: list[str],
    X_test: np.ndarray, y_test: np.ndarray, names_test: list[str],
    target: str,
    train_region: str,
    test_region: str,
    backend: str = "lightgbm",
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    random_state: int = 42,
    max_train_rows: int | None = 400_000,
) -> dict:
    """Train a GBM on ``train_region`` and evaluate on ``test_region``.

    Both matrices are aligned to their common feature set first. ``max_train_rows``
    optionally subsamples the (heavily spatially autocorrelated) training rows for
    speed; set to ``None`` to use all rows.
    """
    feats = common_features(names_train, names_test)
    Xtr = align_features(X_train, names_train, feats)
    Xte = align_features(X_test, names_test, feats)

    if max_train_rows is not None and len(Xtr) > max_train_rows:
        rng = np.random.default_rng(random_state)
        sel = rng.choice(len(Xtr), size=max_train_rows, replace=False)
        Xtr, ytr = Xtr[sel], y_train[sel]
    else:
        ytr = y_train

    model = _make_regressor(backend, n_estimators, learning_rate, random_state)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    m = _metrics(y_test, pred)
    log.info(
        "transfer %s: %s -> %s  R²=%.3f RMSE=%.2f (n=%d, %d feats)",
        target, train_region, test_region, m["r2"], m["rmse"], m["n"], len(feats),
    )
    return {
        "target": target,
        "train_region": train_region,
        "test_region": test_region,
        "n_common_features": len(feats),
        "common_features": feats,
        "backend": backend,
        "out_of_region": m,
    }
