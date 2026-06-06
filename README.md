# SAR-Based Soil Characterization

A geospatial intelligence framework that characterizes soil surface conditions and seasonal moisture behavior using **Sentinel-1 SAR**, **Copernicus DEM**, terrain/hydrological analysis, and machine learning — evolving from exploratory analysis (V1) to a full soil intelligence platform (V4).

**Pilot area:** Konya agricultural plain, Türkiye (~23 × 20 km)  
**Data source:** Sentinel Hub CDSE Processing API (no raw downloads)

---

## Roadmap

| Version | Theme | Status |
|---|---|---|
| **V1** | Exploratory soil characterization | ✅ Complete |
| **V2** | Soil property modelling (clay, SOC, sand via ML) | 📋 Planned |
| **V3** | GeoAI — time-series SAR + deep learning | 📋 Planned |
| **V4** | Interactive soil intelligence platform | 📋 Planned |

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
│   ├── aoi/konya.yml              # AOI bounding box + season dates
│   └── pipelines/v1.yml           # Pipeline parameters
├── src/soilgeo/
│   ├── acquisition/sentinel_hub.py  # Sentinel Hub API client (S1 + DEM)
│   ├── sar/evalscripts.py           # JS evalscripts (VV+VH median, DEM)
│   ├── terrain/derivatives.py       # gdaldem wrappers
│   ├── hydrology/whitebox.py        # WhiteboxTools (TWI, SPI)
│   ├── indices/sar.py               # VV/VH ratio, NDDI
│   ├── analysis/
│   │   ├── classification.py        # k-means Surface Response Classes
│   │   ├── statistics.py            # Kruskal-Wallis, Spearman
│   │   └── report.py                # Stats analysis orchestrator
│   ├── products/cog.py              # COG writer, catalog, risk index
│   └── utils/                       # logging, config, geo helpers
├── pipelines/run_v1.py              # CLI entry point (resumable, stage-by-stage)
├── notebooks/
│   ├── 01_v1_exploratory_report.ipynb
│   └── visualize_v1.py
└── tests/unit/                      # 27 unit tests
```

---

## Tech stack

| Layer | Libraries |
|---|---|
| SAR / EO | Sentinel Hub Python SDK, Copernicus CDSE |
| Terrain | GDAL / gdaldem |
| Hydrology | WhiteboxTools |
| Raster I/O | rasterio, rioxarray, xarray |
| Geospatial | geopandas, shapely, pyproj |
| ML / Stats | scikit-learn, scipy |
| Testing | pytest, ruff |

---

## License

MIT
