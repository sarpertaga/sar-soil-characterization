"""
U-Net training loop, metrics, and MLflow tracking (V3-F5, V3-F7; V3-T4, T5).

Metric & loss conventions (V3-T4):
* classification → weighted cross-entropy loss; reported with **mean IoU** and
  **per-class F1**,
* regression → MSE loss; reported with RMSE/MAE for comparability with V2.

Dual-scale evaluation (V3-T5): because labels derive from 250 m SoilGrids, the
task is effectively label super-resolution. :func:`aggregate_to_scale`
block-averages a 10 m prediction up to the 250 m grid so the honest metric can
be reported alongside the 10 m visual output.

The training utilities are device-agnostic (CPU / CUDA / MPS) so the same code
runs locally and on a Colab/Kaggle GPU (V3-T6). The pure metric helpers are
unit-tested; :func:`train_one_epoch` is exercised by a tiny overfit smoke test.
"""
from __future__ import annotations

import numpy as np

from soilgeo.utils.logging import get_logger

log = get_logger(__name__)

NODATA = -9999.0


# ── Metrics (pure, testable) ────────────────────────────────────────────────

def mean_iou(pred, target, n_classes: int) -> float:
    """Mean Intersection-over-Union over classes present in the union."""
    import torch

    pred = pred.flatten()
    target = target.flatten()
    ious = []
    for c in range(n_classes):
        p = pred == c
        t = target == c
        inter = float((p & t).sum())
        union = float((p | t).sum())
        if union > 0:
            ious.append(inter / union)
    return float(torch.tensor(ious).mean()) if ious else 0.0


def per_class_f1(pred, target, n_classes: int) -> list[float]:
    """Per-class F1 score."""
    pred = pred.flatten()
    target = target.flatten()
    scores = []
    for c in range(n_classes):
        tp = float(((pred == c) & (target == c)).sum())
        fp = float(((pred == c) & (target != c)).sum())
        fn = float(((pred != c) & (target == c)).sum())
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom > 0 else 0.0)
    return scores


def aggregate_to_scale(arr: np.ndarray, factor: int, nodata: float = NODATA) -> np.ndarray:
    """
    Block-average a 2-D raster by an integer factor, ignoring nodata. Used to
    aggregate 10 m predictions up to the 250 m SoilGrids grid (V3-T5). The array
    is trimmed to a multiple of ``factor`` before reshaping.
    """
    h, w = arr.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    a = arr[:h2, :w2].astype(np.float64)
    blocks = a.reshape(h2 // factor, factor, w2 // factor, factor)
    mask = blocks != nodata
    with np.errstate(invalid="ignore"):
        summed = np.where(mask, blocks, 0.0).sum(axis=(1, 3))
        counts = mask.sum(axis=(1, 3))
        out = np.where(counts > 0, summed / np.maximum(counts, 1), nodata)
    return out.astype(np.float32)


# ── Training ────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, task: str, device,
                    class_weights=None, nodata: float = NODATA) -> float:
    """
    Run one training epoch over ``loader`` (iterable of ``(x, y)`` batches).
    Returns the mean batch loss. ``task`` selects the head/loss used.

    Regression uses a **nodata-masked MSE** so padded edge pixels and missing
    labels (``nodata``) never contribute to the loss. Classification uses
    weighted cross-entropy with ``ignore_index`` for nodata classes.
    """
    import torch.nn as nn

    model.train()
    model.to(device)
    ce = None
    if task == "classification":
        ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        if task == "classification":
            loss = ce(out["class_logits"], y.long())
        else:
            pred = out["regression"]
            mask = y != nodata
            if mask.sum() == 0:
                continue
            loss = ((pred - y)[mask] ** 2).mean()      # nodata-masked MSE
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


def evaluate_regression(model, loader, device, nodata: float = NODATA) -> dict:
    """
    Run a regression U-Net over ``loader`` and compute pixel-wise R²/RMSE/MAE on
    valid (non-nodata) pixels — the held-out-split metric comparable to the GBM
    baseline. Returns ``{r2, rmse, mae, n}`` plus the flattened arrays for
    dual-scale aggregation.
    """
    import numpy as np
    import torch
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    model.eval()
    model.to(device)
    preds, tgts = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))["regression"].cpu().numpy()
            yb = y.numpy()
            m = yb != nodata
            preds.append(out[m])
            tgts.append(yb[m])
    p = np.concatenate(preds)
    t = np.concatenate(tgts)
    return {
        "r2": float(r2_score(t, p)),
        "rmse": float(np.sqrt(mean_squared_error(t, p))),
        "mae": float(mean_absolute_error(t, p)),
        "n": int(t.size),
    }


def train_unet(
    model,
    train_loader,
    val_loader,
    *,
    task: str,
    epochs: int,
    lr: float,
    weight_decay: float = 1e-4,
    device: str = "auto",
    class_weights=None,
    mlflow_run=None,
    seed: int = 42,
) -> dict:
    """
    Full training loop with optional MLflow logging (V3-F7). Returns a dict of
    final metrics. Kept dependency-light so it runs identically on a laptop
    (CPU/MPS) and a Colab/Kaggle GPU. MLflow is used only if ``mlflow_run`` is
    an active run context (caller owns ``mlflow.start_run``).
    """
    import torch

    from soilgeo.models.unet import resolve_device

    torch.manual_seed(seed)
    dev = resolve_device(device)
    log.info("Training U-Net: task=%s epochs=%d device=%s", task, epochs, dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, task, dev, class_weights)
        history.append(train_loss)
        log.info("  epoch %d/%d  train_loss=%.4f", epoch + 1, epochs, train_loss)
        if mlflow_run is not None:
            import mlflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)

    return {"task": task, "epochs": epochs, "final_train_loss": history[-1], "history": history}
