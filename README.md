# SAR-Based Soil Characterization

A geospatial intelligence framework that characterizes soil surface conditions and seasonal moisture behavior using **Sentinel-1 SAR**, **Copernicus DEM**, terrain/hydrological analysis, and machine learning — evolving from exploratory analysis (V1) through GeoAI modelling (V3) to independent validation (V4).

**Pilot areas:** Konya plain, Büyük Menderes valley, Göller Yöresi — Türkiye  
**Data source:** Sentinel Hub CDSE Processing API (no raw downloads)

---

## Roadmap

| Version | Theme | Status |
|---|---|---|
| **V1** | Exploratory soil characterization | ✅ Complete |
| **V2** | Soil property modelling (clay, SOC, sand via ML) | ✅ Complete |
| **V3** | GeoAI — time-series SAR + deep learning | ✅ Complete |
| **V4** | Independent validation & generalization (transfer, uncertainty, in-situ) | 🔶 In progress |
| **V5** | Interactive soil intelligence platform | 📋 Planned |

---

## V1 — Exploratory Soil Characterization

### What it does

V1 builds a full analysis-ready raster stack from SAR and terrain data, computes seasonal moisture indices, runs statistical validation, and classifies the landscape into surface response zones.

```
Sentinel Hub API
  ├── Sentinel-1 IW VV+VH σ⁰  (wet season median: Jan–Mar 2024)
  ├── Sentinel-1 IW VV+VH σ⁰  (dry season median: Jul–Sep 2024)
  └── Copernicus DEM GLO-30    (elevation, 30 m)
           ↓
  Terrain:    slope · aspect · curvature · roughness · hillshade
  Hydrology:  TWI · SPI · flow accumulation  (WhiteboxTools)
  Indices:    VV/VH ratio · NDDI
  Analysis:   Kruskal-Wallis · Spearman · k-means classification
           ↓
  Outputs (data/processed/)
  ├── surface_response_classes_konya.tif
  ├── construction_risk_konya.tif
  ├── stats_v1.json
  └── catalog.json
```

### Key results (Konya pilot)

| Metric | Value |
|---|---|
| VV backscatter wet season | −12.97 dB (mean) |
| VV backscatter dry season | −12.79 dB (mean) |
| NDDI wet-dominant pixels | 41.3 % |
| NDDI dry-dominant pixels | 44.6 % |
| Terrain slope (mean / p90) | 1.05° / 2.14° — extremely flat plain |
| KW: VV wet by TWI quartile | H = 9 676, p < 0.001 ✓ |
| KW: NDDI by TWI quartile | H = 4 172, p < 0.001 ✓ |
| Spearman VV wet vs dry | ρ = +0.61 |

### Pipeline stages

```
fetch_dem   → Copernicus DEM from Sentinel Hub
fetch_s1    → Sentinel-1 wet + dry median composites
terrain     → slope, aspect, curvature, roughness, hillshade
hydrology   → TWI (slope floor 0.1°), SPI, flow accumulation
indices     → VV/VH ratio, NDDI
classify    → Surface Response Classes (k-means on NDDI + TWI + slope)
stats       → Kruskal-Wallis + Spearman analysis → stats_v1.json
risk        → Construction Moisture Risk Index
catalog     → product catalog.json
```

### NDDI formula

```
σ_linear = 10^(VV_dB / 10)
NDDI = (σ_wet − σ_dry) / (σ_wet + σ_dry)   ∈ [−1, 1]
```

Positive → wet season backscatter dominant (soil moisture / winter crop)  
Negative → dry season backscatter dominant (summer crop volume scattering / bare soil)

---

## V2 — Soil Property Modelling

### What it does

V2 extends V1 by adding Sentinel-2 optical features and SoilGrids ground truth
to train Random Forest models that predict soil properties at 10 m resolution.

```
V1 feature stack (S1 VV/VH/NDDI + TWI + slope + curvature)
  +
Sentinel-2 L2A bare-soil composite (Jul–Sep 2024, cloud-masked, NDVI < 0.4)
  → BSI · Clay Index · NDVI · NDWI · Iron Oxide
  +
SoilGrids v2.0 (clay · sand · SOC at 250 m)
           ↓
  Feature matrix (7 079 pixels × 13 features)
  Random Forest regression — 5-fold CV
           ↓
  Outputs (data/processed/)
  ├── clay_10m_konya.tif      (g/kg)
  ├── sand_10m_konya.tif      (g/kg)
  ├── soc_10m_konya.tif       (dg/kg)
  └── v2_metrics.json
```

### Key results

| Target | R² | RMSE | Top feature |
|---|---|---|---|
| Clay | 0.49 | 13.98 g/kg | Iron Oxide (0.22) |
| Sand | 0.43 | 8.28 g/kg | BSI (0.19) |
| SOC | 0.23 | 29.42 dg/kg | VV wet (0.11) |

**Konya soil profile:** ~37.7% clay, ~16.7% sand — heavy lacustrine clay plain.  
Iron Oxide ratio (S2 B04/B02) is the strongest predictor for clay content,
consistent with the iron-rich, reddish Konya plain soils.

### Pipeline stages

```
fetch_s2    → Sentinel-2 L2A bare-soil composite (3×3 tiled fetch)
s2_indices  → BSI, Clay Index, NDVI, NDWI, Iron Oxide
soilgrids   → Download clay/sand/SOC from ISRIC WCS
features    → Aggregate features to 250 m, build training matrix
train       → Random Forest (200 trees, 5-fold CV) per target
predict     → Apply model at 10 m, write GeoTIFFs
catalog     → Update product catalog
```

### Run V2 pipeline

```bash
python pipelines/run_v2.py
# or stage by stage:
python pipelines/run_v2.py --stages fetch_s2,s2_indices
python pipelines/run_v2.py --stages soilgrids,features,train,predict
```

### Open the report

```bash
jupyter notebook notebooks/02_v2_soil_modelling_report.ipynb
```

---

## V3 — GeoAI Soil Intelligence

### What it does

V3 extends V2 with **multi-temporal Sentinel-1 features** (per-pixel temporal
statistics + Quegan & Yu speckle filtering), auxiliary **S2 seasonal NDVI/NDMI**,
terrain/hydrology, and optional **climate** (SPI-3), then trains a **GBM strong
baseline** and a **U-Net** — tracked and compared head-to-head on a spatially
held-out test set (spec §7). Runs via the `soilgeo-dl` environment.

```
S1 time series (≥1 yr) → Quegan&Yu filter → temporal stats (mean/std/p10/p90/amplitude/wet-dry)
  + S2 seasonal NDVI/NDMI + terrain (slope/curvature/TWI) + climate (SPI-3)
           ↓  build_cube → feature matrix + spatial-block groups (Zarr)
  GBM (LightGBM, spatial-block GroupKFold)   ← strong baseline
  U-Net (encoder-decoder, classification + regression heads, MPS/GPU)
           ↓
  clay/sand/soc maps · behaviour clusters · risk zones · v3_*_metrics.json
```

### Key result — pilot AOI matters, and DL is not always justified

Two pilots, identical pipeline:

| AOI | SoilGrids CV | GBM clay R² (spatial-CV) |
|---|---|---|
| Konya plain (homogeneous) | ~4 % | **−0.14** (no generalizable signal) |
| Büyük Menderes valley (valley↔slope) | ~25 % | **+0.51** |

Per-target GBM spatial-CV on the Menderes cube (17 features, ~3.1 M samples):
clay **0.43**, sand **0.49**, SOC **0.61** — topography (*slope*) is the top
feature for all three.

**DL-vs-GBM verdict (Menderes clay, held-out, single-geometry S1):** GBM
**0.43** vs U-Net **0.56**. An earlier run scored the U-Net at 0.08 and
concluded the GBM wins — that number turned out to be two engineering bugs, not
a modelling truth: (a) NODATA sentinels (−9999) leaked through normalization
into the convolutions/BatchNorm, and (b) the final-epoch weights were evaluated
instead of the best-on-validation checkpoint (no early stopping). With both
fixed, the U-Net beats the strong baseline by **+0.13 R²** on the same feature
cube, even on this label-super-resolution task. Top GBM feature is *slope* —
topography is the master control on soil texture in relief.

Orbit-geometry control: the original series mixed ascending+descending passes
(the config's `orbit_direction` was silently unused). Re-fetching
ASCENDING-only dropped the GBM from 0.51 to 0.43 — the mixed series had ~2×
observations per median window, so part of the old "baseline strength" was
denser sampling, not cleaner physics. The single-geometry series is the
methodologically defensible one, and both models above are compared on it.

### Flood showcase — dense labels, clearer DL win

The complementary task (`pipelines/flood_showcase.py`, **Sen1Floods11**
hand-labeled SAR): a **dense pixel-mask** problem where spatial texture is the
signal. Same U-Net code:

| Model | mIoU | F1 (water) |
|---|---|---|
| GBM (per-pixel) | 0.712 | 0.663 |
| **U-Net (spatial)** | **0.735** | **0.712** |

*(446 chips, 256×256; numbers from `data/processed_flood/flood_showcase_metrics.json`.)*

**Lesson:** the margin DL buys tracks the label quality and the spatial nature
of the signal — clear on dense hand-labeled masks (flood), narrow on coarse
250 m super-resolved labels (soil). And before ruling against a model, make
sure it lost fairly: the soil U-Net's first "defeat" was a data-pipeline bug.

### Run V3

```bash
conda activate soilgeo-dl          # see environment-dl.yml
KMP_DUPLICATE_LIB_OK=TRUE python pipelines/run_v3.py \
  --config config/pipelines/v3_menderes.yml --stages all
# flood showcase:
KMP_DUPLICATE_LIB_OK=TRUE python pipelines/flood_showcase.py
# interactive maps:
KMP_DUPLICATE_LIB_OK=TRUE python pipelines/show_maps.py
```

Reports: `notebooks/03_v3_geoai_report.ipynb`, `notebooks/04_flood_unet_showcase.ipynb`

---

## V4 — Independent Validation & Generalization

V1–V3 are trained *and* scored against SoilGrids 250 m — a model product. A
spatial-CV R² therefore measures agreement with another model, not with the
ground. V4 asks the three questions a soil-mapping product actually faces, and
reports the answers honestly — including the negative ones.

### 1. Does a model transfer to another region? — **No.**

A GBM trained on one region's full feature matrix, evaluated on the other
(common 17-feature set, `soilgeo/analysis/transfer.py`):

| Train → Test | R² | R² (bias-corrected) | Spearman ρ | Bias (g/kg) |
|---|---|---|---|---|
| Menderes → Konya | −22.7 | 0.01 | 0.06 | −59.4 |
| Konya → Menderes | −2.9 | 0.01 | 0.04 | +123.7 |

Even after removing the constant offset, out-of-region correlation is ~0: the
learned signal is **region-specific**. In-region spatial-CV (0.43) says nothing
about out-of-region skill — a per-region calibration (or region covariates +
multi-region training) is a hard requirement for any scaled-up product.

### 2. Are the uncertainty intervals honest? — **Slightly overconfident.**

Quantile GBM (5–50–95 %) with out-of-fold spatial-block predictions
(`v3_uncertainty_metrics.json`, Menderes):

| Target | Nominal coverage | Empirical coverage | Mean interval width |
|---|---|---|---|
| Clay | 90 % | 85.2 % | 108.5 g/kg |
| Sand | 90 % | 85.5 % | 137.3 g/kg |
| SOC | 90 % | 85.3 % | 287.7 dg/kg |

A ~5 pt under-coverage on unseen blocks — usable, but intervals should be
inflated (e.g. conformal calibration) before being shown to a user.

### 3. Does the model beat its own label source on real soil? — **Pending.**

The `validate_insitu` stage (`soilgeo/analysis/validation.py`) scores a
predicted map against **21 real WoSIS soil profiles** (topsoil clay/sand/SOC,
CC BY 4.0) held out entirely from training — and scores **SoilGrids itself at
the same points**, so the model is judged relative to its label source. The
Göller Yöresi pilot AOI (Isparta–Burdur, strong texture gradient: clay 2–86 %)
and the validation CSV are configured and ready
(`config/pipelines/v3_goller.yml`, `data/validation/wosis_goller_insitu.csv`);
the fetch stages await Sentinel Hub quota.

```bash
KMP_DUPLICATE_LIB_OK=TRUE python pipelines/run_v3.py \
  --config config/pipelines/v3_goller.yml --stages all
```

---

## Quickstart

### Prerequisites

- [miniforge3](https://github.com/conda-forge/miniforge) (conda + mamba)
- [Sentinel Hub CDSE account](https://dataspace.copernicus.eu/) with OAuth client credentials

### Setup

```bash
git clone https://github.com/sarpertaga/sar-soil-characterization
cd sar-soil-characterization

mamba env create -f environment.yml
conda activate soilgeo
pip install -e .

cp .env.example .env
# Edit .env — add your SH_CLIENT_ID and SH_CLIENT_SECRET
```

### Run V1 pipeline

```bash
# Full pipeline
python pipelines/run_v1.py

# Specific stages
python pipelines/run_v1.py --stages fetch_dem,fetch_s1
python pipelines/run_v1.py --stages indices,classify,stats,catalog
```

### Open the report

```bash
jupyter notebook notebooks/01_v1_exploratory_report.ipynb
```

---

## Project structure

```
sar-soil-characterization/
├── config/
│   ├── aoi/                           # AOI bbox + seasons (konya, menderes, göller)
│   └── pipelines/                     # v1 / v2 / v3_{menderes,goller,pilot} params
├── src/soilgeo/
│   ├── acquisition/
│   │   ├── sentinel_hub.py            # Sentinel Hub API client (S1 + DEM)
│   │   ├── sentinel2.py               # S2 L2A bare-soil fetch (3×3 tiled)
│   │   ├── s1_timeseries.py           # S1 time-series fetch (per-orbit medians)
│   │   ├── climate.py                 # ERA5-Land / CHIRPS climate series
│   │   └── soilgrids.py               # SoilGrids v2.0 WCS downloader
│   ├── sar/
│   │   ├── evalscripts.py             # JS evalscripts (S1, DEM, S2 bare-soil)
│   │   └── multitemporal.py           # Quegan & Yu multi-temporal speckle filter
│   ├── terrain/derivatives.py         # gdaldem wrappers
│   ├── hydrology/whitebox.py          # WhiteboxTools (TWI, SPI, flow accum.)
│   ├── indices/                       # SAR (VV/VH, NDDI) + optical (BSI, NDVI, …)
│   ├── features/                      # temporal stats + SPI-3 climate features
│   ├── modelling/                     # V2 feature matrix + Random Forest CV
│   ├── models/
│   │   ├── cube.py                    # V3 feature-cube assembly (Zarr)
│   │   ├── gbm.py                     # LightGBM spatial-CV + quantile uncertainty
│   │   ├── unet.py / train.py         # U-Net + training loop (early stopping, MLflow)
│   │   ├── dataset.py                 # tile dataset, train-only norm stats, augment
│   │   └── tiling.py                  # spatial-block tiling (GroupKFold ids)
│   ├── analysis/
│   │   ├── classification.py          # k-means Surface Response Classes
│   │   ├── statistics.py / report.py  # Kruskal-Wallis, Spearman orchestration
│   │   ├── transfer.py                # V4 cross-region transferability
│   │   └── validation.py              # V4 in-situ (WoSIS) validation vs SoilGrids
│   ├── products/cog.py                # COG writer, catalog, risk index
│   └── utils/                         # logging, config, geo helpers
├── pipelines/
│   ├── run_v1.py / run_v2.py / run_v3.py  # resumable stage-by-stage CLIs
│   ├── flood_showcase.py              # Sen1Floods11 GBM-vs-U-Net showcase
│   └── show_maps.py                   # interactive product maps
├── notebooks/
│   ├── 01_v1_exploratory_report.ipynb # V1 analysis report
│   ├── 02_v2_soil_modelling_report.ipynb # V2 ML report + feature importance
│   ├── 03_v3_geoai_report.ipynb       # V3 DL-vs-GBM verdict + product maps
│   ├── 04_flood_unet_showcase.ipynb   # flood showcase report
│   └── cdse_start.ipynb               # CDSE JupyterHub — cloud-native execution
├── data/validation/                   # WoSIS in-situ points (V4, CC BY 4.0)
└── tests/unit/                        # 113 unit tests (soilgeo-dl env; DL tests skip in base env)
```

---

## Tech stack

| Layer | Libraries |
|---|---|
| SAR / EO | Sentinel Hub Python SDK, Copernicus CDSE |
| Optical | Sentinel-2 L2A (CDSE Processing API) |
| Labels / in-situ | SoilGrids v2.0 WCS (ISRIC) · WoSIS soil profiles (V4 validation) |
| Terrain | GDAL / gdaldem |
| Hydrology | WhiteboxTools |
| Raster I/O | rasterio, rioxarray, xarray |
| Geospatial | geopandas, shapely, pyproj |
| ML / Stats | scikit-learn, scipy |
| Notebooks | Jupyter, matplotlib |
| Testing | pytest, ruff |

---

## License

MIT
