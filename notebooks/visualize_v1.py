"""Quick V1 results visualization."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio

INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("V1 — Konya SAR Soil Characterization", fontsize=14, fontweight="bold")

def read(path, band=1):
    with rasterio.open(path) as src:
        data = src.read(band).astype(np.float32)
        nd = src.nodata or -9999.0
    data[data == nd] = np.nan
    return data

# ── Row 1: SAR ───────────────────────────────────────────────────────────────
vv_wet = read(INTERIM / "s1/s1_vv_konya_wet.tif")
ax = axes[0, 0]
im = ax.imshow(vv_wet, cmap="gray", vmin=-25, vmax=-5)
plt.colorbar(im, ax=ax, label="dB")
ax.set_title("VV Backscatter — Wet Season")
ax.axis("off")

vv_dry = read(INTERIM / "s1/s1_vv_konya_dry.tif")
ax = axes[0, 1]
im = ax.imshow(vv_dry, cmap="gray", vmin=-25, vmax=-5)
plt.colorbar(im, ax=ax, label="dB")
ax.set_title("VV Backscatter — Dry Season")
ax.axis("off")

nddi = read(INTERIM / "indices/nddi.tif")
ax = axes[0, 2]
im = ax.imshow(nddi, cmap="RdBu", vmin=-0.3, vmax=0.3)
plt.colorbar(im, ax=ax, label="-1→1")
ax.set_title("NDDI (wet−dry seasonal)")
ax.axis("off")

ratio = read(INTERIM / "indices/vv_vh_ratio_wet.tif")
ax = axes[0, 3]
im = ax.imshow(ratio, cmap="RdYlGn_r", vmin=3, vmax=12)
plt.colorbar(im, ax=ax, label="dB")
ax.set_title("VV/VH Ratio (Wet)")
ax.axis("off")

# ── Row 2: Terrain + Classification ──────────────────────────────────────────
twi = read(INTERIM / "hydrology/twi.tif")
ax = axes[1, 0]
im = ax.imshow(twi, cmap="terrain", vmin=np.nanpercentile(twi, 2), vmax=np.nanpercentile(twi, 98))
plt.colorbar(im, ax=ax)
ax.set_title("Topographic Wetness Index")
ax.axis("off")

slope = read(INTERIM / "terrain/slope.tif")
ax = axes[1, 1]
im = ax.imshow(slope, cmap="hot_r", vmin=0, vmax=15)
plt.colorbar(im, ax=ax, label="°")
ax.set_title("Slope")
ax.axis("off")

classes = read(PROCESSED / "surface_response_classes_konya.tif")
cmap = mcolors.ListedColormap(["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)
ax = axes[1, 2]
im = ax.imshow(classes, cmap=cmap, norm=norm)
cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4])
cbar.ax.set_yticklabels(["Dry", "Mod. Dry", "Trans.", "Seas. Wet", "Wet"], fontsize=8)
ax.set_title("Surface Response Classes")
ax.axis("off")

risk = read(PROCESSED / "construction_risk_konya.tif")
ax = axes[1, 3]
im = ax.imshow(risk, cmap="RdYlGn_r", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label="0→1")
ax.set_title("Construction Moisture Risk")
ax.axis("off")

plt.tight_layout()
out = PROCESSED / "v1_overview.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
