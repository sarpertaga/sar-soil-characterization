#!/usr/bin/env python
"""
Interactive matplotlib viewer for V3 soil products + the flood showcase.
Opens GUI windows (zoom/pan). Run:

    KMP_DUPLICATE_LIB_OK=TRUE python pipelines/show_maps.py
"""
import matplotlib.pyplot as plt
import numpy as np
import rasterio

P = "data/processed_menderes"
F = "data/processed_flood"


def rd(name):
    a = rasterio.open(f"{P}/{name}.tif").read(1).astype(float)
    a[a == -9999] = np.nan
    a[a == 255] = np.nan
    return a


# ── Figure 1: V3 Menderes soil products ──
fig1, ax = plt.subplots(1, 3, figsize=(15, 5))
fig1.canvas.manager.set_window_title("V3 — Menderes soil products (GBM)")
im0 = ax[0].imshow(rd("clay_10m_menderes_pilot"), cmap="YlOrBr")
ax[0].set_title("Clay (g/kg) — GBM 10 m")
plt.colorbar(im0, ax=ax[0], shrink=.7)
im1 = ax[1].imshow(rd("behaviour_clusters_menderes_pilot"), cmap="tab10")
ax[1].set_title("Soil-behaviour clusters")
plt.colorbar(im1, ax=ax[1], shrink=.7)
im2 = ax[2].imshow(rd("risk_zones_menderes_pilot"), cmap="Reds")
ax[2].set_title("Erosion risk")
plt.colorbar(im2, ax=ax[2], shrink=.7)
for a in ax:
    a.axis("off")
fig1.tight_layout()

# ── Figure 2: flood showcase GBM vs U-Net ──
ex = np.load(f"{F}/flood_examples.npz")
n = min(3, ex["vv"].shape[0])
fig2, ax2 = plt.subplots(n, 4, figsize=(13, 3.2 * n))
fig2.canvas.manager.set_window_title("Flood showcase — GBM vs U-Net")
titles = ["S1 VV (dB)", "ground truth", "GBM (per-pixel)", "U-Net (spatial)"]
for r in range(n):
    t = np.where(ex["truth"][r] == 255, np.nan, ex["truth"][r])
    ax2[r, 0].imshow(ex["vv"][r], cmap="gray")
    ax2[r, 1].imshow(t, cmap="Blues", vmin=0, vmax=1)
    ax2[r, 2].imshow(ex["gbm"][r], cmap="Blues", vmin=0, vmax=1)
    ax2[r, 3].imshow(ex["unet"][r], cmap="Blues", vmin=0, vmax=1)
    for c in range(4):
        ax2[r, c].axis("off")
        if r == 0:
            ax2[r, c].set_title(titles[c])
fig2.tight_layout()

plt.show()
