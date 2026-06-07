#!/usr/bin/env python
"""
Flood-segmentation showcase: where deep learning *does* beat the baseline.

Companion to the V3 soil result (where a per-pixel GBM beat the U-Net because the
labels were 250 m SoilGrids). Here the task is **dense-mask SAR flood mapping**
(Sen1Floods11 hand-labeled): per-pixel ground truth where spatial texture
(flood extent, connectivity) genuinely matters — the regime where a U-Net should
win. We train both models on identical chip splits and compare mean-IoU / F1.

    conda activate soilgeo-dl
    KMP_DUPLICATE_LIB_OK=TRUE python pipelines/flood_showcase.py

Inputs: data/sen1floods11/{S1Hand,LabelHand}/  (see soilgeo.acquisition.sen1floods11)
Output: data/processed_flood/flood_showcase_metrics.json
"""
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from soilgeo.utils.logging import get_logger

log = get_logger("flood_showcase")

DATA = Path("data/sen1floods11")
OUT = Path("data/processed_flood")
SIZE = 256          # chips read at 256×256 (downsampled from 512) for tractable MPS training
IGNORE = 255        # nodata class for CE / metrics
SEED = 42


def _chip_pairs() -> list[tuple[Path, Path]]:
    s1 = {p.name.replace("_S1Hand.tif", ""): p for p in (DATA / "S1Hand").glob("*.tif")}
    lab = {p.name.replace("_LabelHand.tif", ""): p for p in (DATA / "LabelHand").glob("*.tif")}
    keys = sorted(set(s1) & set(lab))
    return [(s1[k], lab[k]) for k in keys]


def _read_s1(path: Path) -> np.ndarray:
    """Read VV+VH at SIZE×SIZE (average-resampled), dB, NaN/inf → 0."""
    with rasterio.open(path) as src:
        a = src.read(out_shape=(2, SIZE, SIZE), resampling=Resampling.average).astype(np.float32)
    a[~np.isfinite(a)] = 0.0
    return a  # [2, SIZE, SIZE]


def _read_label(path: Path) -> np.ndarray:
    """Read flood mask at SIZE×SIZE (nearest); -1 nodata → IGNORE; values {0,1}."""
    with rasterio.open(path) as src:
        a = src.read(1, out_shape=(SIZE, SIZE), resampling=Resampling.nearest)
    out = a.astype(np.int64)
    out[a < 0] = IGNORE
    return out  # [SIZE, SIZE]


def _split(pairs, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_tr, n_va = int(0.7 * len(pairs)), int(0.15 * len(pairs))
    return (idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:])


def _metrics(pred, target):
    """Water-class F1 + mean IoU over {not-water, water}, ignoring IGNORE pixels."""
    m = target != IGNORE
    p, t = pred[m], target[m]
    ious = []
    for c in (0, 1):
        inter = np.sum((p == c) & (t == c))
        union = np.sum((p == c) | (t == c))
        if union > 0:
            ious.append(inter / union)
    tp = np.sum((p == 1) & (t == 1))
    fp = np.sum((p == 1) & (t == 0))
    fn = np.sum((p == 0) & (t == 1))
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return {"miou": float(np.mean(ious)), "f1_water": float(f1)}


# ── GBM baseline (per-pixel) ─────────────────────────────────────────────────

def run_gbm(pairs, tr, te):
    from lightgbm import LGBMClassifier

    def stack(idxs, cap=400_000):
        X, y = [], []
        for i in idxs:
            s1 = _read_s1(pairs[i][0])
            lab = _read_label(pairs[i][1])
            feats = np.stack([s1[0], s1[1], s1[0] - s1[1]], -1).reshape(-1, 3)  # VV,VH,VV-VH
            yl = lab.reshape(-1)
            m = yl != IGNORE
            X.append(feats[m])
            y.append(yl[m])
        X = np.concatenate(X)
        y = np.concatenate(y)
        if len(y) > cap:
            sel = np.random.default_rng(SEED).choice(len(y), cap, replace=False)
            X, y = X[sel], y[sel]
        return X, y

    Xtr, ytr = stack(tr)
    log.info("GBM train pixels: %d (water %.1f%%)", len(ytr), 100 * ytr.mean())
    clf = LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=SEED,
                         n_jobs=-1, verbose=-1)
    clf.fit(Xtr, ytr)

    preds, tgts = [], []
    for i in te:
        s1 = _read_s1(pairs[i][0])
        lab = _read_label(pairs[i][1])
        feats = np.stack([s1[0], s1[1], s1[0] - s1[1]], -1).reshape(-1, 3)
        preds.append(clf.predict(feats))
        tgts.append(lab.reshape(-1))
    return _metrics(np.concatenate(preds), np.concatenate(tgts)), clf


def gbm_predict_chip(clf, s1):
    """GBM water/not-water prediction for one [2,H,W] chip → [H,W]."""
    feats = np.stack([s1[0], s1[1], s1[0] - s1[1]], -1).reshape(-1, 3)
    return clf.predict(feats).reshape(s1.shape[1], s1.shape[2])


# ── U-Net (spatial) ──────────────────────────────────────────────────────────

def run_unet(pairs, tr, va, te, epochs=20):
    import torch
    from torch.utils.data import DataLoader, Dataset

    from soilgeo.models.train import train_one_epoch
    from soilgeo.models.unet import UNet, resolve_device

    # per-channel norm from train chips
    sums = np.zeros(2)
    sqs = np.zeros(2)
    n = 0
    for i in tr:
        a = _read_s1(pairs[i][0]).reshape(2, -1)
        sums += a.sum(1)
        sqs += (a ** 2).sum(1)
        n += a.shape[1]
    mean = sums / n
    std = np.sqrt(np.maximum(sqs / n - mean ** 2, 1e-6))

    class FloodDS(Dataset):
        def __init__(self, idxs):
            self.idxs = idxs

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, k):
            i = self.idxs[k]
            x = _read_s1(pairs[i][0])
            x = (x - mean[:, None, None]) / std[:, None, None]
            y = _read_label(pairs[i][1])
            return torch.from_numpy(x).float(), torch.from_numpy(y).long()

    dev = resolve_device("auto")
    train_dl = DataLoader(FloodDS(tr), batch_size=8, shuffle=True)
    model = UNet(in_channels=2, base_channels=16, depth=4, n_classes=2, regression=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # class weights — water is rare; upweight it
    w = torch.tensor([1.0, 5.0])
    for e in range(epochs):
        loss = train_one_epoch(model, train_dl, opt, "classification", dev, class_weights=w.to(dev))
        log.info("  U-Net epoch %d/%d loss=%.4f", e + 1, epochs, loss)

    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for x, y in DataLoader(FloodDS(te), batch_size=8):
            logits = model(x.to(dev))["class_logits"].cpu()
            preds.append(logits.argmax(1).numpy().reshape(-1))
            tgts.append(y.numpy().reshape(-1))

    def unet_predict_chip(s1):
        xn = (s1 - mean[:, None, None]) / std[:, None, None]
        with torch.no_grad():
            xt = torch.from_numpy(xn).float().unsqueeze(0).to(dev)
            return model(xt)["class_logits"].argmax(1)[0].cpu().numpy()

    return _metrics(np.concatenate(preds), np.concatenate(tgts)), unet_predict_chip


def main():
    pairs = _chip_pairs()
    log.info("Sen1Floods11 hand-labeled chip pairs: %d", len(pairs))
    tr, va, te = _split(pairs)
    log.info("Split: train=%d val=%d test=%d", len(tr), len(va), len(te))

    gbm_m, clf = run_gbm(pairs, tr, te)
    log.info("GBM (per-pixel):  mIoU=%.3f  F1_water=%.3f", gbm_m["miou"], gbm_m["f1_water"])
    unet_m, unet_predict = run_unet(pairs, tr, va, te)
    log.info("U-Net (spatial):  mIoU=%.3f  F1_water=%.3f", unet_m["miou"], unet_m["f1_water"])

    verdict = "U-Net WINS (texture helps)" if unet_m["miou"] > gbm_m["miou"] else "GBM still better"
    log.info("VERDICT: %s", verdict)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "flood_showcase_metrics.json").write_text(json.dumps(
        {"gbm": gbm_m, "unet": unet_m, "verdict": verdict,
         "n_chips": len(pairs), "size": SIZE}, indent=2))

    # Save a few test-chip examples for the report notebook: VV, truth, GBM, U-Net.
    ex = te[:4]
    np.savez_compressed(
        OUT / "flood_examples.npz",
        vv=np.stack([_read_s1(pairs[i][0])[0] for i in ex]),
        truth=np.stack([_read_label(pairs[i][1]) for i in ex]),
        gbm=np.stack([gbm_predict_chip(clf, _read_s1(pairs[i][0])) for i in ex]),
        unet=np.stack([unet_predict(_read_s1(pairs[i][0])) for i in ex]),
    )
    log.info("Saved example predictions for %d test chips", len(ex))


if __name__ == "__main__":
    main()
