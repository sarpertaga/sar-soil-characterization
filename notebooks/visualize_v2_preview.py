"""
V2 output preview — iki ayrı pencere:
  1. V1 sonuçları (SAR, terrain, sınıflandırma)  → kapat
  2. V2 tahmin haritaları (clay, sand, SOC)       → açılır

Run:
    conda activate soilgeo
    python notebooks/visualize_v2_preview.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import rasterio

ROOT      = Path(__file__).parent.parent
INTERIM   = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
NODATA    = -9999.0

TITLE_COLOR = "#e6edf3"
LABEL_COLOR = "#8b949e"
BG         = "#0d1117"
BG2        = "#161b22"
BLUE       = "#58a6ff"


def load(path: Path, band: int = 1) -> np.ma.MaskedArray:
    with rasterio.open(path) as src:
        data = src.read(band).astype(np.float32)
        nd = src.nodata if src.nodata is not None else NODATA
    return np.ma.masked_where(data == nd, data)


def extent(path: Path) -> list:
    with rasterio.open(path) as src:
        b = src.bounds
    return [b.left, b.right, b.bottom, b.top]


def clip(arr, lo=2, hi=98):
    p_lo, p_hi = np.nanpercentile(arr.compressed(), [lo, hi])
    return np.clip(arr, p_lo, p_hi)


def style(ax, title):
    ax.set_title(title, color=TITLE_COLOR, fontsize=9.5, pad=5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")


def colorbar(fig, im, ax, label=None):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=LABEL_COLOR, labelsize=7)
    if label:
        cb.set_label(label, color=LABEL_COLOR, fontsize=7)


# ═══════════════════════════════════════════════════════════════════════════════
# PENCERE 1 — V1 Sonuçları
# ═══════════════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(18, 15), facecolor=BG)
fig1.suptitle(
    "V1 — Exploratory Soil Characterization  |  Konya (~23×20 km)",
    fontsize=14, color="white", fontweight="bold", y=0.99,
)

section_labels = [
    (0.01, 0.94, "SAR BACKSCATTER  (Sentinel-1)"),
    (0.01, 0.62, "TERRAIN & HYDROLOGY  (Copernicus DEM 30 m)"),
    (0.01, 0.30, "V1 OUTPUTS"),
]
for x, y, txt in section_labels:
    fig1.text(x, y, txt, color=BLUE, fontsize=9, fontweight="bold")


def sp1(row, col):
    return fig1.add_subplot(3, 3, row * 3 + col + 1)


# ── SAR row ──────────────────────────────────────────────────────────────────
vv_wet = load(INTERIM / "s1" / "s1_vv_konya_wet.tif")
vv_dry = load(INTERIM / "s1" / "s1_vv_konya_dry.tif")
nddi   = load(INTERIM / "indices" / "nddi.tif")
ext_s1 = extent(INTERIM / "s1" / "s1_vv_konya_wet.tif")

ax = sp1(0, 0)
colorbar(fig1, ax.imshow(clip(vv_wet), cmap="RdYlGn", extent=ext_s1, origin="upper"), ax)
style(ax, "VV Backscatter — Wet Season (dB)\nJan–Mar 2024 median")

ax = sp1(0, 1)
colorbar(fig1, ax.imshow(clip(vv_dry), cmap="RdYlGn", extent=ext_s1, origin="upper"), ax)
style(ax, "VV Backscatter — Dry Season (dB)\nJul–Sep 2024 median")

ax = sp1(0, 2)
colorbar(fig1,
         ax.imshow(nddi, cmap="RdBu", vmin=-0.4, vmax=0.4, extent=ext_s1, origin="upper"),
         ax, label="← dry  |  wet →")
style(ax, "NDDI  (σ_wet − σ_dry) / (σ_wet + σ_dry)\n41.3% wet-dominant  |  44.6% dry-dominant")

# ── Terrain row ──────────────────────────────────────────────────────────────
slope = load(INTERIM / "terrain" / "slope_resampled.tif")
twi   = load(INTERIM / "hydrology" / "twi_resampled.tif")
fa    = load(INTERIM / "hydrology" / "flow_acc.tif")
ext_t = extent(INTERIM / "terrain" / "slope_resampled.tif")

ax = sp1(1, 0)
colorbar(fig1, ax.imshow(clip(slope, 2, 99), cmap="YlOrBr", extent=ext_t, origin="upper"), ax)
style(ax, "Slope (°)  —  mean 1.05°  p90: 2.14°\nExtremely flat agricultural plain")

ax = sp1(1, 1)
colorbar(fig1, ax.imshow(clip(twi), cmap="Blues", extent=ext_t, origin="upper"), ax)
style(ax, "TWI — Topographic Wetness Index\nln(flow_acc / tan(slope))")

ax = sp1(1, 2)
colorbar(fig1,
         ax.imshow(clip(np.ma.log(1 + fa)), cmap="PuBu", extent=ext_t, origin="upper"), ax)
style(ax, "Flow Accumulation (log scale)\nDrenaj ağı")

# ── V1 outputs row ───────────────────────────────────────────────────────────
classes  = load(PROCESSED / "surface_response_classes_konya.tif")
risk     = load(PROCESSED / "construction_risk_konya.tif")
ext_proc = extent(PROCESSED / "surface_response_classes_konya.tif")

CLASS_COLORS = ["#d73027", "#f46d43", "#fee090", "#74add1", "#313695"]
CLASS_NAMES  = ["Dry", "Moderately Dry", "Transitional", "Seasonally Wet", "Persistently Wet"]

ax = sp1(2, 0)
ax.imshow(classes, cmap=mcolors.ListedColormap(CLASS_COLORS),
          vmin=0, vmax=4, extent=ext_proc, origin="upper")
ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(CLASS_COLORS, CLASS_NAMES)],
          loc="lower left", fontsize=6, facecolor=BG2, edgecolor="#30363d", labelcolor="white")
style(ax, "Surface Response Classes (k-means, k=5)\nNDDI + TWI + Slope")

ax = sp1(2, 1)
colorbar(fig1, ax.imshow(clip(risk), cmap="RdYlGn_r", extent=ext_proc, origin="upper"),
         ax, label="low ← risk → high")
style(ax, "Construction Moisture Risk Index\n0.5×NDDI + 0.3×log(FA) + 0.2×(1−slope)")

ax = sp1(2, 2)
ax.set_facecolor(BG2)
style(ax, "V1 Key Statistics")
ax.text(0.05, 0.95, (
    "Kruskal-Wallis  VV_wet × TWI quartile\n"
    "  H = 9 676    p < 0.001  ✓\n\n"
    "Kruskal-Wallis  NDDI × TWI quartile\n"
    "  H = 4 172    p < 0.001  ✓\n\n"
    "Spearman  VV_wet vs VV_dry\n"
    "  ρ = +0.61    p < 0.001  ✓\n\n"
    "NDDI wet-dominant pixels :  41.3 %\n"
    "NDDI dry-dominant pixels :  44.6 %\n\n"
    "Slope mean / p90 : 1.05° / 2.14°\n"
    "→ Extremely flat agricultural plain"
), transform=ax.transAxes, fontsize=8.5, color=TITLE_COLOR,
   va="top", fontfamily="monospace", linespacing=1.65)

fig1.tight_layout(rect=[0, 0, 1, 0.97])


# ═══════════════════════════════════════════════════════════════════════════════
# PENCERE 2 — V2 Soil Property Predictions
# ═══════════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(18, 7), facecolor=BG)
fig2.suptitle(
    "V2 — Soil Property Predictions  |  Konya 10 m  (Random Forest · S1 + S2 + Terrain → SoilGrids)",
    fontsize=13, color="white", fontweight="bold", y=1.01,
)

V2_TARGETS = [
    ("clay", "Clay Content  (g/kg → %)", "YlOrBr", True),
    ("sand", "Sand Content  (g/kg → %)", "OrRd",   True),
    ("soc",  "SOC — Soil Organic Carbon  (dg/kg)", "YlGn", False),
]

for col, (prop, title, cmap, convert_pct) in enumerate(V2_TARGETS):
    ax = fig2.add_subplot(1, 3, col + 1)
    pred_path = PROCESSED / f"{prop}_10m_konya.tif"

    if pred_path.exists():
        data = load(pred_path)
        if convert_pct:
            data = data / 10.0
        colorbar(fig2, ax.imshow(clip(data), cmap=cmap,
                                 extent=ext_proc, origin="upper"), ax)
    else:
        ax.set_facecolor(BG2)
        ax.text(0.5, 0.5, (
            f"Pipeline henüz çalıştırılmadı\n\n"
            f"python pipelines/run_v2.py\n\n"
            f"Çıktı : {prop}_10m_konya.tif\n"
            f"Çözünürlük : 10 m\n"
            f"Model : RF 200 ağaç, 5-fold CV\n\n"
            f"Features (13 adet):\n"
            f"  S1  : VV/VH wet+dry, NDDI\n"
            f"  DEM : slope, TWI, curvature\n"
            f"  S2  : BSI, clay_idx, NDVI,\n"
            f"        NDWI, iron_oxide"
        ), transform=ax.transAxes, fontsize=9, color=BLUE,
           ha="center", va="center", fontfamily="monospace", linespacing=1.7,
           bbox=dict(boxstyle="round,pad=0.6", facecolor=BG, edgecolor=BLUE, alpha=0.85))

    style(ax, title)

fig2.tight_layout()

# İki pencereyi sırayla göster
plt.show()
