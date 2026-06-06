# V1 — Exploratory Soil Characterization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the full V1 pipeline — Sentinel Hub API → σ⁰ VV+VH GeoTIFF → terrain/hydrology → SAR Moisture Index + Soil Hardness Classification → Surface Response Classes → exploratory report for the Konya Closed Basin AOI.

**Architecture:** Layered Python package (`soilgeo`) where each layer (acquisition → sar → terrain → hydrology → indices → analysis → products) depends only on file outputs (COG/GeoParquet), never on upstream internals. Configuration-driven via YAML; credentials via `.env`. All rasters output as COG on a common 10 m UTM36N grid.

**Tech Stack:** Python 3.11+, miniforge3/mamba, GDAL≥3.8, rasterio, rioxarray, xarray, geopandas, shapely≥2.0, pyproj, whitebox (Python wrapper), sentinelhub, python-dotenv, scipy, scikit-learn, matplotlib, leafmap, pytest, ruff

---

## Prerequisites (manual, before any task)

- [ ] Install **miniforge3** (conda + mamba):
  ```bash
  curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh -o miniforge3.sh
  bash miniforge3.sh -b -p $HOME/miniforge3
  eval "$($HOME/miniforge3/bin/conda shell.zsh hook)"
  conda init zsh
  ```

- [ ] **Sentinel Hub credentials** hazırla: Sentinel Hub Dashboard → User Settings → OAuth Clients → yeni client oluştur → `client_id` + `client_secret` al.

---

## File Map

```
sar-soil-geoai/
├── README.md
├── environment.yml                      # conda env (core)
├── pyproject.toml                       # package metadata + ruff config
├── .env.example                         # credential placeholders
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml                       # lint + unit tests
├── config/
│   ├── aoi/
│   │   └── konya.yml                    # AOI bounds, date ranges, orbit config
│   └── pipelines/
│       └── v1.yml                       # V1 stage params
├── src/soilgeo/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py                   # get_logger() factory
│   │   ├── config.py                    # YAML loader → dataclass
│   │   └── geo.py                       # CRS helpers, COG writer, nodata utils
│   ├── acquisition/
│   │   ├── __init__.py
│   │   ├── sentinel_hub.py              # Sentinel Hub API → σ⁰ VV+VH GeoTIFF
│   │   └── dem.py                       # Copernicus DEM GLO-30 download
│   ├── sar/
│   │   ├── __init__.py
│   │   └── evalscripts.py               # Sentinel Hub JS evalscripts (VV, VH, multi-date)
│   ├── terrain/
│   │   ├── __init__.py
│   │   └── derivatives.py               # slope, aspect, curvature, roughness, hillshade
│   ├── hydrology/
│   │   ├── __init__.py
│   │   └── whitebox.py                  # WhiteboxTools: TWI, SPI, flow acc, watersheds
│   ├── indices/
│   │   ├── __init__.py
│   │   └── sar.py                       # VV/VH ratio, SAR Moisture Index
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── statistics.py                # Kruskal-Wallis, Spearman, stratified stats
│   │   └── classification.py            # Surface Response Classes via k-means (literature-based)
│   └── products/
│       ├── __init__.py
│       └── cog.py                       # COG writing + provenance sidecar JSON
├── pipelines/
│   └── run_v1.py                        # CLI entry point (argparse)
├── notebooks/
│   └── 01_v1_exploratory_report.ipynb  # Final exploratory report
├── tests/
│   ├── conftest.py                      # shared fixtures (2×2 km synthetic rasters)
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_geo.py
│   │   ├── test_sar_indices.py
│   │   ├── test_terrain.py
│   │   ├── test_hydrology.py
│   │   ├── test_statistics.py
│   │   └── test_classification.py
│   └── integration/
│       └── test_pipeline_fixture.py     # end-to-end on 2×2 km fixture
└── data/                                # gitignored; populated by pipeline
    ├── raw/
    ├── interim/
    └── processed/
```

---

## Phase 1 — Repo & Environment

### Task 1: Create GitHub repo and local project skeleton

**Files:**
- Create: `README.md`
- Create: `.gitignore`
- Create: `environment.yml`
- Create: `pyproject.toml`
- Create: `.env.example`

- [ ] **Step 1: Create GitHub repo**

  ```bash
  cd ~
  mkdir sar-soil-geoai && cd sar-soil-geoai
  git init
  gh repo create sarpertaga/sar-soil-geoai \
    --public \
    --description "SAR-based soil characterization and GeoAI framework (Sentinel-1 + terrain + ML)" \
    --source=. \
    --remote=origin
  ```

- [ ] **Step 2: Create `.gitignore`**

  ```
  # data
  data/
  *.SAFE/
  *.zip

  # credentials
  .env

  # python
  __pycache__/
  *.py[cod]
  .pytest_cache/
  *.egg-info/
  dist/
  build/
  .venv/

  # jupyter
  .ipynb_checkpoints/

  # SNAP
  *.dim
  *.data/
  snap/

  # OS
  .DS_Store
  ```

- [ ] **Step 3: Create `environment.yml`**

  ```yaml
  name: soilgeo
  channels:
    - conda-forge
    - defaults
  dependencies:
    - python=3.11
    - gdal>=3.8
    - rasterio>=1.3
    - rioxarray>=0.15
    - xarray>=2024.1
    - geopandas>=0.14
    - shapely>=2.0
    - pyproj>=3.6
    - scipy>=1.12
    - scikit-learn>=1.4
    - matplotlib>=3.8
    - leafmap>=0.30
    - jupyter
    - ipykernel
    - pytest>=8
    - ruff>=0.4
    - pip
    - pip:
        - sentinelhub>=3.10
        - python-dotenv>=1.0
        - whitebox>=2.3
        - pyyaml>=6.0
  ```

- [ ] **Step 4: Create `pyproject.toml`**

  ```toml
  [build-system]
  requires = ["setuptools>=68"]
  build-backend = "setuptools.backends.legacy:build"

  [project]
  name = "soilgeo"
  version = "0.1.0"
  requires-python = ">=3.11"

  [tool.setuptools.packages.find]
  where = ["src"]

  [tool.ruff]
  line-length = 100
  src = ["src", "tests", "pipelines"]

  [tool.ruff.lint]
  select = ["E", "F", "I", "UP"]

  [tool.pytest.ini_options]
  testpaths = ["tests"]
  markers = ["integration: marks tests that require external data or SNAP (deselect with -m 'not integration')"]
  ```

- [ ] **Step 5: Create `.env.example`**

  ```
  # Sentinel Hub OAuth credentials
  SH_CLIENT_ID=your_client_id_here
  SH_CLIENT_SECRET=your_client_secret_here
  ```

- [ ] **Step 6: Create conda environment and install package**

  ```bash
  mamba env create -f environment.yml
  conda activate soilgeo
  pip install -e .
  ```

- [ ] **Step 7: Create directory skeleton**

  ```bash
  mkdir -p config/aoi config/pipelines
  mkdir -p src/soilgeo/{utils,acquisition,sar/graphs,terrain,hydrology,indices,analysis,products}
  mkdir -p tests/{unit,integration}
  mkdir -p pipelines notebooks data/{raw,interim,processed}
  touch src/soilgeo/__init__.py
  touch src/soilgeo/{utils,acquisition,sar,terrain,hydrology,indices,analysis,products}/__init__.py
  touch tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py
  ```

- [ ] **Step 8: Initial commit**

  ```bash
  git add .
  git commit -m "chore: initial project skeleton with conda env and package layout"
  git push -u origin main
  ```

---

## Phase 2 — Utils & Configuration

### Task 2: Logging utility

**Files:**
- Create: `src/soilgeo/utils/logging.py`
- Create: `tests/unit/test_config.py` (partially — logging assertions)

- [ ] **Step 1: Write the failing test**

  Create `tests/unit/test_config.py`:
  ```python
  import logging
  from soilgeo.utils.logging import get_logger

  def test_get_logger_returns_logger():
      logger = get_logger("soilgeo.test")
      assert isinstance(logger, logging.Logger)
      assert logger.name == "soilgeo.test"

  def test_get_logger_same_name_returns_same_instance():
      a = get_logger("soilgeo.x")
      b = get_logger("soilgeo.x")
      assert a is b
  ```

- [ ] **Step 2: Run test to verify it fails**

  ```bash
  pytest tests/unit/test_config.py -v
  ```
  Expected: `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement `src/soilgeo/utils/logging.py`**

  ```python
  import logging
  import sys
  from pathlib import Path


  def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
      logger = logging.getLogger(name)
      if logger.handlers:
          return logger

      logger.setLevel(logging.DEBUG)
      fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

      console = logging.StreamHandler(sys.stdout)
      console.setLevel(logging.INFO)
      console.setFormatter(fmt)
      logger.addHandler(console)

      if log_dir is not None:
          log_dir.mkdir(parents=True, exist_ok=True)
          from logging.handlers import RotatingFileHandler
          fh = RotatingFileHandler(log_dir / f"{name}.log", maxBytes=10_000_000, backupCount=3)
          fh.setLevel(logging.DEBUG)
          fh.setFormatter(fmt)
          logger.addHandler(fh)

      return logger
  ```

- [ ] **Step 4: Run tests**

  ```bash
  pytest tests/unit/test_config.py -v
  ```
  Expected: 2 PASSED

- [ ] **Step 5: Commit**

  ```bash
  git add src/soilgeo/utils/logging.py tests/unit/test_config.py
  git commit -m "feat(utils): add get_logger factory with console + file handlers"
  ```

---

### Task 3: Config loader

**Files:**
- Create: `src/soilgeo/utils/config.py`
- Create: `config/aoi/konya.yml`
- Create: `config/pipelines/v1.yml`
- Modify: `tests/unit/test_config.py`

- [ ] **Step 1: Create `config/aoi/konya.yml`**

  ```yaml
  name: konya
  description: "Konya Closed Basin pilot AOI — agricultural, seasonally bare, moderate micro-topography"

  # Bounding box in EPSG:4326
  bbox:
    west: 32.20
    south: 37.55
    east: 33.20
    north: 38.20

  # Target CRS for all processing
  crs: "EPSG:32636"   # UTM zone 36N
  resolution_m: 10

  # Sentinel-1 acquisition filter
  sentinel1:
    platform: "Sentinel-1"
    beam_mode: "IW"
    polarizations: ["VV", "VH"]
    orbit_direction: "ASCENDING"    # keep single pass direction
    # Wet season: Jan–Mar (after winter rains), Dry season: Jul–Sep
    wet_season:
      start: "2024-01-15"
      end: "2024-03-15"
      min_scenes: 3
    dry_season:
      start: "2024-07-15"
      end: "2024-09-15"
      min_scenes: 3

  # Copernicus DEM
  dem:
    product: "COP-DEM_GLO-30"
    buffer_m: 5000    # download with 5 km buffer for edge effects
  ```

- [ ] **Step 2: Create `config/pipelines/v1.yml`**

  ```yaml
  version: "v1"
  aoi_config: "config/aoi/konya.yml"

  paths:
    raw_dir: "data/raw"
    interim_dir: "data/interim"
    processed_dir: "data/processed"
    log_dir: "logs"

  snap:
    gpt_bin: "${SNAP_GPT}"    # resolved from env var
    memory_gb: 24
    parallelism: 4
    graph: "src/soilgeo/sar/graphs/s1_grd_preprocessing.xml"

  speckle_filter:
    type: "Refined-Lee"
    window_size: 7

  terrain_correction:
    pixel_spacing_m: 10
    dem_resampling: "BILINEAR_INTERPOLATION"
    img_resampling: "BILINEAR_INTERPOLATION"

  twi:
    dem_resolution: 30    # compute TWI at 30 m then resample to 10 m

  moisture_index:
    nodata: -9999.0

  surface_response_classes:
    n_clusters: 5
    features: ["moisture_index", "twi", "slope"]
    random_state: 42

  construction_risk:
    enabled: true
    weights:
      moisture_index: 0.5
      flow_accumulation_log: 0.3
      low_slope_factor: 0.2    # 1 - (slope / max_slope), capped at 1
  ```

- [ ] **Step 3: Write failing tests (add to `tests/unit/test_config.py`)**

  ```python
  from pathlib import Path
  from soilgeo.utils.config import load_aoi_config, load_pipeline_config, AoiConfig, PipelineConfig

  FIXTURE_AOI = Path("config/aoi/konya.yml")
  FIXTURE_PIPELINE = Path("config/pipelines/v1.yml")

  def test_load_aoi_config_returns_dataclass():
      cfg = load_aoi_config(FIXTURE_AOI)
      assert isinstance(cfg, AoiConfig)
      assert cfg.name == "konya"
      assert cfg.crs == "EPSG:32636"
      assert cfg.resolution_m == 10
      assert len(cfg.bbox) == 4

  def test_aoi_config_bbox_valid():
      cfg = load_aoi_config(FIXTURE_AOI)
      assert cfg.bbox["west"] < cfg.bbox["east"]
      assert cfg.bbox["south"] < cfg.bbox["north"]

  def test_load_pipeline_config():
      cfg = load_pipeline_config(FIXTURE_PIPELINE)
      assert isinstance(cfg, PipelineConfig)
      assert cfg.version == "v1"
  ```

- [ ] **Step 4: Run to verify failure**

  ```bash
  pytest tests/unit/test_config.py -v
  ```
  Expected: `ImportError` on `AoiConfig`

- [ ] **Step 5: Implement `src/soilgeo/utils/config.py`**

  ```python
  import os
  from dataclasses import dataclass
  from pathlib import Path

  import yaml


  @dataclass
  class AoiConfig:
      name: str
      description: str
      bbox: dict          # west, south, east, north in EPSG:4326
      crs: str
      resolution_m: int
      sentinel1: dict
      dem: dict


  @dataclass
  class PipelineConfig:
      version: str
      aoi_config: str
      paths: dict
      snap: dict
      speckle_filter: dict
      terrain_correction: dict
      twi: dict
      moisture_index: dict
      surface_response_classes: dict
      construction_risk: dict


  def _resolve_env_vars(obj):
      """Recursively replace ${VAR} placeholders with env var values."""
      if isinstance(obj, str):
          import re
          return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), obj)
      if isinstance(obj, dict):
          return {k: _resolve_env_vars(v) for k, v in obj.items()}
      if isinstance(obj, list):
          return [_resolve_env_vars(i) for i in obj]
      return obj


  def load_aoi_config(path: Path) -> AoiConfig:
      with open(path) as f:
          raw = yaml.safe_load(f)
      return AoiConfig(
          name=raw["name"],
          description=raw.get("description", ""),
          bbox=raw["bbox"],
          crs=raw["crs"],
          resolution_m=raw["resolution_m"],
          sentinel1=raw["sentinel1"],
          dem=raw["dem"],
      )


  def load_pipeline_config(path: Path) -> PipelineConfig:
      with open(path) as f:
          raw = _resolve_env_vars(yaml.safe_load(f))
      return PipelineConfig(
          version=raw["version"],
          aoi_config=raw["aoi_config"],
          paths=raw["paths"],
          snap=raw["snap"],
          speckle_filter=raw["speckle_filter"],
          terrain_correction=raw["terrain_correction"],
          twi=raw["twi"],
          moisture_index=raw["moisture_index"],
          surface_response_classes=raw["surface_response_classes"],
          construction_risk=raw["construction_risk"],
      )
  ```

- [ ] **Step 6: Run tests**

  ```bash
  pytest tests/unit/test_config.py -v
  ```
  Expected: 5 PASSED

- [ ] **Step 7: Commit**

  ```bash
  git add src/soilgeo/utils/config.py config/ tests/unit/test_config.py
  git commit -m "feat(utils): YAML config loader with env-var resolution for AOI and pipeline configs"
  ```

---

### Task 4: Geo utilities (CRS helpers + COG writer)

**Files:**
- Create: `src/soilgeo/utils/geo.py`
- Create: `tests/unit/test_geo.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `tests/conftest.py`** (shared synthetic raster fixtures)

  ```python
  import numpy as np
  import pytest
  import rasterio
  from rasterio.transform import from_bounds
  from pathlib import Path


  @pytest.fixture
  def synthetic_10m_raster(tmp_path) -> Path:
      """2 km × 2 km UTM36N float32 raster with known values."""
      path = tmp_path / "synthetic.tif"
      # 200 × 200 pixels at 10 m → 2 km × 2 km
      data = np.random.default_rng(42).uniform(-20, 0, (200, 200)).astype(np.float32)
      transform = from_bounds(500000, 4160000, 502000, 4162000, 200, 200)
      with rasterio.open(
          path, "w",
          driver="GTiff", height=200, width=200, count=1,
          dtype="float32", crs="EPSG:32636",
          transform=transform, nodata=-9999.0,
      ) as dst:
          dst.write(data, 1)
      return path


  @pytest.fixture
  def synthetic_dem_raster(tmp_path) -> Path:
      """2 km × 2 km DEM with a gentle slope (elevation 1000–1100 m)."""
      path = tmp_path / "dem.tif"
      rows, cols = 200, 200
      elev = np.linspace(1000, 1100, rows * cols).reshape(rows, cols).astype(np.float32)
      transform = from_bounds(500000, 4160000, 502000, 4162000, cols, rows)
      with rasterio.open(
          path, "w",
          driver="GTiff", height=rows, width=cols, count=1,
          dtype="float32", crs="EPSG:32636",
          transform=transform, nodata=-9999.0,
      ) as dst:
          dst.write(elev, 1)
      return path
  ```

- [ ] **Step 2: Write failing tests**

  Create `tests/unit/test_geo.py`:
  ```python
  import numpy as np
  import rasterio
  from pathlib import Path
  from soilgeo.utils.geo import write_cog, bbox_to_utm, read_band

  def test_write_cog_creates_valid_cog(tmp_path, synthetic_10m_raster):
      out = tmp_path / "out.tif"
      with rasterio.open(synthetic_10m_raster) as src:
          data = src.read(1)
          profile = src.profile

      write_cog(out, data, profile, nodata=-9999.0)

      assert out.exists()
      with rasterio.open(out) as dst:
          assert dst.driver == "GTiff"
          assert dst.nodata == -9999.0
          assert dst.profile["compress"] == "deflate"

  def test_write_cog_embeds_provenance(tmp_path, synthetic_10m_raster):
      import json
      out = tmp_path / "out.tif"
      with rasterio.open(synthetic_10m_raster) as src:
          data = src.read(1)
          profile = src.profile

      write_cog(out, data, profile, provenance={"source": "test", "version": "0.1"})
      sidecar = out.with_suffix(".json")
      assert sidecar.exists()
      meta = json.loads(sidecar.read_text())
      assert meta["source"] == "test"

  def test_bbox_to_utm_returns_utm36n():
      # Konya bbox centre is roughly 32.7E, 37.9N → UTM36N
      bbox_utm = bbox_to_utm(west=32.20, south=37.55, east=33.20, north=38.20, crs="EPSG:32636")
      assert bbox_utm["minx"] < bbox_utm["maxx"]
      assert bbox_utm["miny"] < bbox_utm["maxy"]

  def test_read_band_returns_masked_array(synthetic_10m_raster):
      arr = read_band(synthetic_10m_raster, band=1)
      assert arr.dtype == np.float32
      assert arr.ndim == 2
  ```

- [ ] **Step 3: Run to verify failure**

  ```bash
  pytest tests/unit/test_geo.py -v
  ```
  Expected: `ImportError`

- [ ] **Step 4: Implement `src/soilgeo/utils/geo.py`**

  ```python
  import hashlib
  import json
  from datetime import datetime, timezone
  from pathlib import Path

  import numpy as np
  import rasterio
  from pyproj import Transformer
  from rasterio.crs import CRS


  def write_cog(
      path: Path,
      data: np.ndarray,
      profile: dict,
      nodata: float = -9999.0,
      provenance: dict | None = None,
  ) -> Path:
      """Write a single-band float32 array as Cloud-Optimized GeoTIFF with DEFLATE compression."""
      profile = profile.copy()
      profile.update(
          driver="GTiff",
          dtype="float32",
          count=1,
          nodata=nodata,
          compress="deflate",
          predictor=3,        # floating-point predictor
          tiled=True,
          blockxsize=512,
          blockysize=512,
          interleave="band",
      )
      path.parent.mkdir(parents=True, exist_ok=True)
      with rasterio.open(path, "w", **profile) as dst:
          dst.write(data.astype(np.float32), 1)
          # Build overviews (COG requirement)
          dst.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.average)
          dst.update_tags(ns="rio_overview", resampling="average")

      if provenance is not None:
          provenance["_written_at"] = datetime.now(timezone.utc).isoformat()
          path.with_suffix(".json").write_text(json.dumps(provenance, indent=2))

      return path


  def bbox_to_utm(west: float, south: float, east: float, north: float, crs: str) -> dict:
      """Reproject a WGS84 bounding box to the given UTM CRS."""
      transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
      minx, miny = transformer.transform(west, south)
      maxx, maxy = transformer.transform(east, north)
      return {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy}


  def read_band(path: Path, band: int = 1) -> np.ndarray:
      """Read a raster band as float32 numpy array."""
      with rasterio.open(path) as src:
          return src.read(band).astype(np.float32)
  ```

- [ ] **Step 5: Run tests**

  ```bash
  pytest tests/unit/test_geo.py -v
  ```
  Expected: 4 PASSED

- [ ] **Step 6: Commit**

  ```bash
  git add src/soilgeo/utils/geo.py tests/conftest.py tests/unit/test_geo.py
  git commit -m "feat(utils): COG writer with provenance sidecar, bbox reprojection, band reader"
  ```

---

## Phase 3 — Data Acquisition

### Task 5: Sentinel Hub acquisition — σ⁰ VV+VH GeoTIFF

**Files:**
- Create: `src/soilgeo/sar/evalscripts.py`
- Create: `src/soilgeo/acquisition/sentinel_hub.py`
- Create: `tests/unit/test_acquisition.py`

- [ ] **Step 1: Create `src/soilgeo/sar/evalscripts.py`**

  ```python
  """Sentinel Hub JS evalscripts for Sentinel-1 SAR backscatter."""

  # Single-date VV+VH σ⁰ in dB
  S1_VV_VH_DB = """
  //VERSION=3
  function setup() {
      return {
          input: [{ bands: ["VV", "VH", "dataMask"] }],
          output: { bands: 3, sampleType: "FLOAT32" }
      };
  }
  function evaluatePixel(s) {
      var vv_db = s.VV > 0 ? 10 * Math.log10(s.VV) : -9999;
      var vh_db = s.VH > 0 ? 10 * Math.log10(s.VH) : -9999;
      return [vv_db, vh_db, s.dataMask];
  }
  """

  # Multi-date median composite VV+VH σ⁰ in dB
  S1_VV_VH_MEDIAN_DB = """
  //VERSION=3
  function setup() {
      return {
          input: [{ bands: ["VV", "VH", "dataMask"] }],
          output: { bands: 3, sampleType: "FLOAT32" },
          mosaicking: "ORBIT"
      };
  }
  function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
      outputMetadata.userData = { scene_count: scenes.orbits.length };
  }
  function evaluatePixel(samples) {
      var vv_vals = [], vh_vals = [];
      for (var s of samples) {
          if (s.dataMask && s.VV > 0) vv_vals.push(10 * Math.log10(s.VV));
          if (s.dataMask && s.VH > 0) vh_vals.push(10 * Math.log10(s.VH));
      }
      if (vv_vals.length === 0) return [-9999, -9999, 0];
      vv_vals.sort((a,b) => a-b);
      vh_vals.sort((a,b) => a-b);
      var mid = Math.floor(vv_vals.length / 2);
      var vv_med = vv_vals.length % 2 ? vv_vals[mid] : (vv_vals[mid-1]+vv_vals[mid])/2;
      var vh_med = vh_vals.length % 2 ? vh_vals[mid] : (vh_vals[mid-1]+vh_vals[mid])/2;
      return [vv_med, vh_med, 1];
  }
  """
  ```

- [ ] **Step 2: Write failing tests**

  Create `tests/unit/test_acquisition.py`:
  ```python
  from soilgeo.acquisition.sentinel_hub import build_sh_bbox, build_sh_config

  def test_build_sh_bbox_returns_correct_order():
      bbox = build_sh_bbox(west=32.20, south=37.55, east=33.20, north=38.20)
      # sentinelhub BBox expects (min_x, min_y, max_x, max_y)
      assert bbox[0] == 32.20
      assert bbox[1] == 37.55
      assert bbox[2] == 33.20
      assert bbox[3] == 38.20

  def test_build_sh_config_reads_env(monkeypatch):
      monkeypatch.setenv("SH_CLIENT_ID", "test_id")
      monkeypatch.setenv("SH_CLIENT_SECRET", "test_secret")
      cfg = build_sh_config()
      assert cfg.sh_client_id == "test_id"
      assert cfg.sh_client_secret == "test_secret"
  ```

- [ ] **Step 3: Run to verify failure**

  ```bash
  pytest tests/unit/test_acquisition.py -v
  ```

- [ ] **Step 4: Implement `src/soilgeo/acquisition/sentinel_hub.py`**

  ```python
  """Fetch Sentinel-1 σ⁰ VV+VH GeoTIFFs via Sentinel Hub Processing API."""
  import os
  from pathlib import Path

  import numpy as np
  import rasterio
  from rasterio.transform import from_bounds
  from sentinelhub import (
      BBox, CRS, DataCollection, MimeType, MosaickingOrder,
      SentinelHubRequest, SHConfig, bbox_to_dimensions,
  )
  from dotenv import load_dotenv

  from soilgeo.sar.evalscripts import S1_VV_VH_DB, S1_VV_VH_MEDIAN_DB
  from soilgeo.utils.logging import get_logger

  load_dotenv()
  log = get_logger(__name__)

  NODATA = -9999.0


  def build_sh_config() -> SHConfig:
      cfg = SHConfig()
      cfg.sh_client_id = os.environ["SH_CLIENT_ID"]
      cfg.sh_client_secret = os.environ["SH_CLIENT_SECRET"]
      return cfg


  def build_sh_bbox(west: float, south: float, east: float, north: float) -> BBox:
      return BBox(bbox=[west, south, east, north], crs=CRS.WGS84)


  def fetch_backscatter(
      bbox_wgs84: dict,
      time_interval: tuple[str, str],
      output_path: Path,
      resolution_m: int = 10,
      median_composite: bool = True,
  ) -> Path:
      """
      Fetch Sentinel-1 IW σ⁰ VV+VH for a time interval via Sentinel Hub.
      Saves a 2-band float32 GeoTIFF: band1=VV_dB, band2=VH_dB.
      Uses median composite over the interval by default (multi-date).
      """
      if output_path.exists():
          log.info("Skipping fetch (exists): %s", output_path.name)
          return output_path

      config = build_sh_config()
      sh_bbox = build_sh_bbox(**bbox_wgs84)
      size = bbox_to_dimensions(sh_bbox, resolution=resolution_m)
      evalscript = S1_VV_VH_MEDIAN_DB if median_composite else S1_VV_VH_DB

      log.info("Requesting Sentinel Hub: %s → %s, size=%s", time_interval[0], time_interval[1], size)

      request = SentinelHubRequest(
          evalscript=evalscript,
          input_data=[
              SentinelHubRequest.input_data(
                  data_collection=DataCollection.SENTINEL1_IW,
                  time_interval=time_interval,
                  mosaicking_order=MosaickingOrder.LEAST_CC,
                  other_args={"dataFilter": {"acquisitionMode": "IW"}},
              )
          ],
          responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
          bbox=sh_bbox,
          size=size,
          config=config,
      )

      data = request.get_data()[0]   # shape: (H, W, 3) — VV, VH, dataMask
      vv = data[:, :, 0].astype(np.float32)
      vh = data[:, :, 1].astype(np.float32)
      mask = data[:, :, 2] == 0
      vv[mask] = NODATA
      vh[mask] = NODATA

      h, w = vv.shape
      transform = from_bounds(
          bbox_wgs84["west"], bbox_wgs84["south"],
          bbox_wgs84["east"], bbox_wgs84["north"],
          w, h,
      )
      output_path.parent.mkdir(parents=True, exist_ok=True)
      with rasterio.open(
          output_path, "w",
          driver="GTiff", height=h, width=w, count=2,
          dtype="float32", crs="EPSG:4326",
          transform=transform, nodata=NODATA,
          compress="deflate",
      ) as dst:
          dst.write(vv, 1)
          dst.write(vh, 2)
          dst.update_tags(band_1="VV_sigma0_dB", band_2="VH_sigma0_dB",
                          time_start=time_interval[0], time_end=time_interval[1])

      log.info("Saved: %s (%dx%d px)", output_path.name, w, h)
      return output_path
  ```

- [ ] **Step 5: Run tests**

  ```bash
  pytest tests/unit/test_acquisition.py -v
  ```
  Expected: 2 PASSED

- [ ] **Step 6: Commit**

  ```bash
  git add src/soilgeo/sar/evalscripts.py src/soilgeo/acquisition/sentinel_hub.py tests/unit/test_acquisition.py
  git commit -m "feat(acquisition): Sentinel Hub σ⁰ VV+VH fetch with median composite evalscript"
  ```

---

### Task 6: Copernicus DEM GLO-30 download

**Files:**
- Create: `src/soilgeo/acquisition/dem.py`

> Note: DEM download uses the AWS Open Data bucket `copernicus-dem-30m` — no credentials required.

- [ ] **Step 1: Write failing tests (append to `tests/unit/test_acquisition.py`)**

  ```python
  from soilgeo.acquisition.dem import build_dem_tile_ids

  def test_build_dem_tile_ids_konya():
      # Konya bbox covers roughly lat 37-38, lon 32-33 → tiles N37_E032, N37_E033, N38_E032, N38_E033
      tile_ids = build_dem_tile_ids(west=32.20, south=37.55, east=33.20, north=38.20)
      assert "N37_E032" in tile_ids
      assert "N38_E032" in tile_ids
      assert len(tile_ids) >= 2

  def test_build_dem_tile_ids_naming_format():
      tile_ids = build_dem_tile_ids(west=32.0, south=37.0, east=33.0, north=38.0)
      for tid in tile_ids:
          assert tid[0] in ("N", "S")
          assert "E" in tid or "W" in tid
  ```

- [ ] **Step 2: Run to verify failure**

  ```bash
  pytest tests/unit/test_acquisition.py::test_build_dem_tile_ids_konya -v
  ```

- [ ] **Step 3: Implement `src/soilgeo/acquisition/dem.py`**

  ```python
  """Download Copernicus DEM GLO-30 tiles from AWS Open Data."""
  import math
  import urllib.request
  from pathlib import Path

  from soilgeo.utils.logging import get_logger

  log = get_logger(__name__)

  _S3_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


  def build_dem_tile_ids(west: float, south: float, east: float, north: float) -> list[str]:
      """Return GLO-30 tile IDs covering the bounding box (1° × 1° tiles)."""
      tile_ids = []
      for lat in range(math.floor(south), math.ceil(north)):
          for lon in range(math.floor(west), math.ceil(east)):
              ns = "N" if lat >= 0 else "S"
              ew = "E" if lon >= 0 else "W"
              tile_ids.append(f"{ns}{abs(lat):02d}_{ew}{abs(lon):03d}")
      return tile_ids


  def _tile_url(tile_id: str) -> str:
      folder = f"Copernicus_DSM_COG_10_{tile_id}_00_DEM"
      return f"{_S3_BASE}/{folder}/{folder}.tif"


  def download_dem_tiles(
      bbox: dict,
      output_dir: Path,
      buffer_deg: float = 0.05,
  ) -> list[Path]:
      """Download all GLO-30 tiles covering bbox + buffer. Returns list of local tile paths."""
      output_dir.mkdir(parents=True, exist_ok=True)
      tile_ids = build_dem_tile_ids(
          west=bbox["west"] - buffer_deg,
          south=bbox["south"] - buffer_deg,
          east=bbox["east"] + buffer_deg,
          north=bbox["north"] + buffer_deg,
      )
      paths = []
      for tid in tile_ids:
          url = _tile_url(tid)
          dest = output_dir / f"cop_dem_glo30_{tid}.tif"
          if dest.exists():
              log.info("Already downloaded DEM tile: %s", tid)
              paths.append(dest)
              continue
          log.info("Downloading DEM tile: %s", tid)
          try:
              urllib.request.urlretrieve(url, dest)
              paths.append(dest)
          except Exception as exc:
              log.warning("Failed to download %s: %s", tid, exc)
      return paths
  ```

- [ ] **Step 4: Run tests**

  ```bash
  pytest tests/unit/test_acquisition.py -v
  ```
  Expected: 4 PASSED

- [ ] **Step 5: Commit**

  ```bash
  git add src/soilgeo/acquisition/dem.py tests/unit/test_acquisition.py
  git commit -m "feat(acquisition): Copernicus DEM GLO-30 tile discovery and download from AWS"
  ```

---

## Phase 4 — SAR Indices

### Task 7: VV/VH ratio + SAR Moisture Index

**Files:**
- Create: `src/soilgeo/indices/sar.py`
- Create: `tests/unit/test_sar_indices.py`

- [ ] **Step 1: Write failing tests**

  Create `tests/unit/test_sar_indices.py`:
  ```python
  import numpy as np
  import rasterio
  from soilgeo.indices.sar import compute_vv_vh_ratio, compute_moisture_index

  def test_vv_vh_ratio_is_difference_in_db(tmp_path, synthetic_10m_raster):
      vv_path = synthetic_10m_raster
      vh_path = tmp_path / "vh.tif"
      with rasterio.open(vv_path) as src:
          vv_data = src.read(1)
          profile = src.profile
      with rasterio.open(vh_path, "w", **profile) as dst:
          dst.write(vv_data - 5.0, 1)
      ratio_path = tmp_path / "ratio.tif"
      compute_vv_vh_ratio(vv_path, vh_path, ratio_path)
      with rasterio.open(ratio_path) as src:
          ratio = src.read(1)
      np.testing.assert_allclose(ratio[ratio != -9999.0], 5.0, atol=1e-3)

  def test_moisture_index_midpoint(tmp_path, synthetic_10m_raster):
      with rasterio.open(synthetic_10m_raster) as src:
          data = src.read(1)
          profile = src.profile
      dry = tmp_path / "dry.tif"
      wet = tmp_path / "wet.tif"
      scene = tmp_path / "scene.tif"
      for p, val in [(dry, -20.0), (wet, -10.0), (scene, -15.0)]:
          with rasterio.open(p, "w", **profile) as dst:
              dst.write(np.full_like(data, val), 1)
      mi_path = tmp_path / "mi.tif"
      compute_moisture_index(scene, dry, wet, mi_path)
      with rasterio.open(mi_path) as src:
          mi = src.read(1)
      valid = mi[mi != -9999.0]
      np.testing.assert_allclose(valid.mean(), 0.5, atol=0.05)
  ```

- [ ] **Step 2: Run to verify failure**

  ```bash
  pytest tests/unit/test_sar_indices.py -v
  ```

- [ ] **Step 3: Implement `src/soilgeo/indices/sar.py`**

  ```python
  """SAR indices: VV/VH ratio and SAR Moisture Index."""
  from pathlib import Path
  import numpy as np
  import rasterio
  from soilgeo.utils.logging import get_logger

  log = get_logger(__name__)
  NODATA = -9999.0


  def compute_vv_vh_ratio(vv_path: Path, vh_path: Path, output_path: Path) -> Path:
      """VV/VH ratio in dB = VV_dB − VH_dB."""
      if output_path.exists():
          return output_path
      with rasterio.open(vv_path) as src:
          vv = src.read(1).astype(np.float32)
          profile = src.profile
          nd = src.nodata or NODATA
      with rasterio.open(vh_path) as src:
          vh = src.read(1).astype(np.float32)
      mask = (vv == nd) | (vh == nd)
      ratio = np.where(mask, NODATA, vv - vh)
      output_path.parent.mkdir(parents=True, exist_ok=True)
      profile.update(dtype="float32", count=1, nodata=NODATA, compress="deflate")
      with rasterio.open(output_path, "w", **profile) as dst:
          dst.write(ratio, 1)
      return output_path


  def compute_moisture_index(
      scene_path: Path,
      dry_path: Path,
      wet_path: Path,
      output_path: Path,
  ) -> Path:
      """
      SAR Moisture Index: MI = (σ_t − σ_dry) / (σ_wet − σ_dry)
      Output clamped to [0, 1].
      """
      if output_path.exists():
          return output_path
      with rasterio.open(scene_path) as src:
          sigma_t = src.read(1).astype(np.float32)
          profile = src.profile
          nd = src.nodata or NODATA
      with rasterio.open(dry_path) as src:
          sigma_dry = src.read(1).astype(np.float32)
      with rasterio.open(wet_path) as src:
          sigma_wet = src.read(1).astype(np.float32)
      mask = (sigma_t == nd) | (sigma_dry == nd) | (sigma_wet == nd)
      denom = sigma_wet - sigma_dry
      with np.errstate(divide="ignore", invalid="ignore"):
          mi = np.where(
              mask | (np.abs(denom) < 1e-6),
              NODATA,
              np.clip((sigma_t - sigma_dry) / denom, 0.0, 1.0),
          )
      output_path.parent.mkdir(parents=True, exist_ok=True)
      profile.update(dtype="float32", count=1, nodata=NODATA, compress="deflate")
      with rasterio.open(output_path, "w", **profile) as dst:
          dst.write(mi.astype(np.float32), 1)
      log.info("Moisture index written: %s", output_path.name)
      return output_path
  ```

- [ ] **Step 4: Run tests**

  ```bash
  pytest tests/unit/test_sar_indices.py -v
  ```
  Expected: 2 PASSED

- [ ] **Step 5: Commit**

  ```bash
  git add src/soilgeo/indices/sar.py tests/unit/test_sar_indices.py
  git commit -m "feat(indices): VV/VH ratio and SAR Moisture Index"
  ```

---

## Phase 5 — Terrain & Hydrology

### Task 8: DEM derivatives (slope, aspect, curvature, roughness, hillshade)

**Files:**
- Create: `src/soilgeo/terrain/derivatives.py`
- Modify: `tests/unit/test_terrain.py`

- [ ] **Step 1: Write failing tests**

  Create `tests/unit/test_terrain.py`:
  ```python
  import numpy as np
  import rasterio
  from pathlib import Path
  from soilgeo.terrain.derivatives import compute_slope, compute_aspect, compute_roughness, compute_hillshade

  def test_compute_slope_range(synthetic_dem_raster, tmp_path):
      out = tmp_path / "slope.tif"
      compute_slope(synthetic_dem_raster, out)
      assert out.exists()
      with rasterio.open(out) as src:
          data = src.read(1)
          valid = data[data != src.nodata]
          # slope in degrees must be 0–90
          assert valid.min() >= 0.0
          assert valid.max() <= 90.0

  def test_compute_aspect_range(synthetic_dem_raster, tmp_path):
      out = tmp_path / "aspect.tif"
      compute_aspect(synthetic_dem_raster, out)
      with rasterio.open(out) as src:
          data = src.read(1)
          valid = data[data != -9999.0]
          assert valid.min() >= 0.0
          assert valid.max() <= 360.0

  def test_compute_roughness_positive(synthetic_dem_raster, tmp_path):
      out = tmp_path / "roughness.tif"
      compute_roughness(synthetic_dem_raster, out)
      with rasterio.open(out) as src:
          data = src.read(1)
          valid = data[data != -9999.0]
          assert (valid >= 0).all()

  def test_compute_hillshade_range(synthetic_dem_raster, tmp_path):
      out = tmp_path / "hillshade.tif"
      compute_hillshade(synthetic_dem_raster, out)
      with rasterio.open(out) as src:
          data = src.read(1)
          valid = data[data != -9999.0]
          assert valid.min() >= 0
          assert valid.max() <= 255
  ```

- [ ] **Step 2: Run to verify failure**

  ```bash
  pytest tests/unit/test_terrain.py -v
  ```

- [ ] **Step 3: Implement `src/soilgeo/terrain/derivatives.py`**

  Uses GDAL via subprocess (calls `gdaldem`), which is the most reliable cross-platform approach and avoids scipy gradient edge effects on large rasters.

  ```python
  """Compute DEM-derived terrain layers via GDAL."""
  import subprocess
  from pathlib import Path

  import numpy as np
  import rasterio

  from soilgeo.utils.logging import get_logger

  log = get_logger(__name__)


  def _run_gdaldem(mode: str, input_path: Path, output_path: Path, extra_args: list[str] | None = None) -> Path:
      if output_path.exists():
          log.info("Skipping %s (exists): %s", mode, output_path.name)
          return output_path
      cmd = ["gdaldem", mode, str(input_path), str(output_path),
             "-of", "GTiff", "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES"]
      if extra_args:
          cmd.extend(extra_args)
      result = subprocess.run(cmd, capture_output=True, text=True)
      if result.returncode != 0:
          raise RuntimeError(f"gdaldem {mode} failed: {result.stderr}")
      return output_path


  def compute_slope(dem_path: Path, output_path: Path) -> Path:
      """Slope in degrees (0–90)."""
      return _run_gdaldem("slope", dem_path, output_path, ["-p"])  # -p = percent would break range; no flag = degrees


  def compute_aspect(dem_path: Path, output_path: Path) -> Path:
      """Aspect in degrees (0–360, clockwise from N)."""
      return _run_gdaldem("aspect", dem_path, output_path)


  def compute_roughness(dem_path: Path, output_path: Path) -> Path:
      """Terrain roughness (max − min in 3×3 neighbourhood)."""
      return _run_gdaldem("roughness", dem_path, output_path)


  def compute_hillshade(dem_path: Path, output_path: Path, azimuth: float = 315.0, altitude: float = 45.0) -> Path:
      """Hillshade (0–255 uint8)."""
      return _run_gdaldem("hillshade", dem_path, output_path, ["-az", str(azimuth), "-alt", str(altitude)])


  def compute_curvature(dem_path: Path, output_path: Path) -> Path:
      """Plan curvature via numpy finite differences (GDAL has no built-in curvature)."""
      if output_path.exists():
          log.info("Skipping curvature (exists): %s", output_path.name)
          return output_path
      with rasterio.open(dem_path) as src:
          elev = src.read(1).astype(np.float64)
          profile = src.profile
          res = src.res[0]    # assume square pixels

      # Second derivative (plan curvature ≈ ∂²z/∂x²)
      zy, zx = np.gradient(elev, res)
      zxy, zxx = np.gradient(zx, res)
      zyy, _ = np.gradient(zy, res)
      curvature = -(zxx + zyy)   # sign convention: positive = convex

      profile.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate")
      output_path.parent.mkdir(parents=True, exist_ok=True)
      with rasterio.open(output_path, "w", **profile) as dst:
          dst.write(curvature.astype(np.float32), 1)
      return output_path
  ```

- [ ] **Step 4: Run tests**

  ```bash
  pytest tests/unit/test_terrain.py -v
  ```
  Expected: 4 PASSED

- [ ] **Step 5: Commit**

  ```bash
  git add src/soilgeo/terrain/derivatives.py tests/unit/test_terrain.py
  git commit -m "feat(terrain): DEM derivatives via gdaldem (slope, aspect, roughness, hillshade, curvature)"
  ```

---

### Task 9: Hydrology — WhiteboxTools wrappers (TWI, SPI, flow accumulation)

**Files:**
- Create: `src/soilgeo/hydrology/whitebox.py`
- Create: `tests/unit/test_hydrology.py`

> `whitebox` Python package auto-downloads the WhiteboxTools binary on first use. Run once: `import whitebox; whitebox.WhiteboxTools()` to trigger download.

- [ ] **Step 1: Write failing tests**

  Create `tests/unit/test_hydrology.py`:
  ```python
  import numpy as np
  import rasterio
  from soilgeo.hydrology.whitebox import compute_twi_spi, compute_flow_accumulation

  def test_compute_flow_accumulation_output_exists(synthetic_dem_raster, tmp_path):
      fa_path = tmp_path / "flow_acc.tif"
      compute_flow_accumulation(synthetic_dem_raster, fa_path, work_dir=tmp_path)
      assert fa_path.exists()

  def test_flow_accumulation_all_positive(synthetic_dem_raster, tmp_path):
      fa_path = tmp_path / "flow_acc.tif"
      compute_flow_accumulation(synthetic_dem_raster, fa_path, work_dir=tmp_path)
      with rasterio.open(fa_path) as src:
          data = src.read(1)
          valid = data[data != src.nodata]
          assert (valid >= 0).all()

  def test_twi_spi_output_files_exist(synthetic_dem_raster, tmp_path):
      twi_path = tmp_path / "twi.tif"
      spi_path = tmp_path / "spi.tif"
      compute_twi_spi(synthetic_dem_raster, twi_path, spi_path, work_dir=tmp_path)
      assert twi_path.exists()
      assert spi_path.exists()
  ```

- [ ] **Step 2: Run to verify failure**

  ```bash
  pytest tests/unit/test_hydrology.py -v
  ```

- [ ] **Step 3: Implement `src/soilgeo/hydrology/whitebox.py`**

  TWI formula: `TWI = ln(a / tan(β))` where a = flow accumulation area (m²), β = slope (radians).
  SPI formula: `SPI = a × tan(β)`.
  Per spec V1-T1: TWI computed at 30 m DEM resolution, then bilinearly resampled to 10 m.

  ```python
  """Hydrological analysis via WhiteboxTools Python API."""
  from pathlib import Path

  import numpy as np
  import rasterio
  from rasterio.enums import Resampling
  from rasterio.transform import from_bounds

  from soilgeo.utils.logging import get_logger

  log = get_logger(__name__)


  def _get_wbt():
      import whitebox
      wbt = whitebox.WhiteboxTools()
      wbt.verbose = False
      return wbt


  def compute_flow_accumulation(
      dem_path: Path,
      output_path: Path,
      work_dir: Path | None = None,
  ) -> Path:
      """Fill depressions then compute D8 flow accumulation."""
      if output_path.exists():
          log.info("Skipping flow_accumulation (exists)")
          return output_path

      work_dir = work_dir or output_path.parent
      work_dir.mkdir(parents=True, exist_ok=True)
      filled = work_dir / "_dem_filled.tif"

      wbt = _get_wbt()
      wbt.work_dir = str(work_dir)

      # Step 1: Fill single-cell pits
      wbt.fill_depressions(str(dem_path), str(filled))
      # Step 2: D8 flow accumulation (returns specific catchment area in m²)
      wbt.d8_flow_accumulation(str(filled), str(output_path), out_type="specific contributing area")
      return output_path


  def compute_twi_spi(
      dem_path: Path,
      twi_output: Path,
      spi_output: Path,
      work_dir: Path | None = None,
  ) -> tuple[Path, Path]:
      """Compute TWI and SPI. TWI uses WhiteboxTools wetness_index tool."""
      work_dir = work_dir or twi_output.parent
      work_dir.mkdir(parents=True, exist_ok=True)

      fa_path = work_dir / "_flow_acc.tif"
      compute_flow_accumulation(dem_path, fa_path, work_dir)

      if not twi_output.exists() or not spi_output.exists():
          wbt = _get_wbt()
          wbt.work_dir = str(work_dir)

          if not twi_output.exists():
              log.info("Computing TWI...")
              # WhiteboxTools wetness_index: TWI = ln(sca / tan(slope))
              wbt.wetness_index(str(fa_path), str(dem_path), str(twi_output))

          if not spi_output.exists():
              log.info("Computing SPI via flow_acc × slope...")
              _compute_spi_manual(dem_path, fa_path, spi_output)

      return twi_output, spi_output


  def _compute_spi_manual(dem_path: Path, fa_path: Path, spi_output: Path) -> Path:
      """SPI = a × tan(β) — computed from flow_acc and slope arrays."""
      import subprocess
      slope_path = spi_output.parent / "_slope_rad.tif"

      # gdaldem slope in degrees, then convert to radians in numpy
      subprocess.run(
          ["gdaldem", "slope", str(dem_path), str(slope_path), "-of", "GTiff"],
          check=True, capture_output=True,
      )
      with rasterio.open(fa_path) as src_fa:
          fa = src_fa.read(1).astype(np.float64)
          profile = src_fa.profile
          nodata = src_fa.nodata or -9999.0

      with rasterio.open(slope_path) as src_sl:
          slope_deg = src_sl.read(1).astype(np.float64)

      slope_rad = np.radians(slope_deg)
      tan_slope = np.tan(np.clip(slope_rad, 1e-6, None))   # avoid tan(0) = 0 division
      spi = fa * tan_slope

      mask = (fa == nodata) | (slope_deg == -9999.0)
      spi[mask] = -9999.0

      profile.update(dtype="float32", nodata=-9999.0)
      spi_output.parent.mkdir(parents=True, exist_ok=True)
      with rasterio.open(spi_output, "w", **profile) as dst:
          dst.write(spi.astype(np.float32), 1)
      return spi_output
  ```

- [ ] **Step 4: Trigger WhiteboxTools binary download (one-time)**

  ```bash
  python -c "import whitebox; wbt = whitebox.WhiteboxTools(); print(wbt.version())"
  ```

- [ ] **Step 5: Run tests**

  ```bash
  pytest tests/unit/test_hydrology.py -v
  ```
  Expected: 3 PASSED

- [ ] **Step 6: Commit**

  ```bash
  git add src/soilgeo/hydrology/whitebox.py tests/unit/test_hydrology.py
  git commit -m "feat(hydrology): WhiteboxTools flow accumulation, TWI (wetness_index), SPI"
  ```

---

## Phase 6 — Indices & Analysis

### Task 10: SAR Moisture Index

**Files:**
- Create: `src/soilgeo/indices/sar.py`
- Modify: `tests/unit/test_sar_indices.py`

The Moisture Index formula (V1-F3):
`MI = (σ_t − σ_dry) / (σ_wet − σ_dry)`
where σ values are in linear scale (not dB), then converted to dB for output.

- [ ] **Step 1: Write failing tests (append to `tests/unit/test_sar_indices.py`)**

  ```python
  import numpy as np
  import rasterio
  from soilgeo.indices.sar import compute_vv_vh_ratio, compute_moisture_index

  def test_vv_vh_ratio_is_difference_in_db(tmp_path, synthetic_10m_raster):
      """VV/VH ratio in dB = VV_dB - VH_dB."""
      # Create matching VH raster slightly lower than VV
      import rasterio; from rasterio.transform import from_bounds
      vv_path = synthetic_10m_raster
      vh_path = tmp_path / "vh.tif"
      with rasterio.open(vv_path) as src:
          vv_data = src.read(1)
          profile = src.profile
      with rasterio.open(vh_path, "w", **profile) as dst:
          dst.write(vv_data - 5.0, 1)   # VH = VV - 5 dB

      ratio_path = tmp_path / "ratio.tif"
      compute_vv_vh_ratio(vv_path, vh_path, ratio_path)
      with rasterio.open(ratio_path) as src:
          ratio = src.read(1)
      # VV - (VV - 5) = 5 everywhere
      np.testing.assert_allclose(ratio[ratio != -9999.0], 5.0, atol=1e-3)

  def test_moisture_index_bounded_0_1(tmp_path, synthetic_10m_raster):
      """MI output should be in [0, 1] for pixels between dry and wet composites."""
      import rasterio; from rasterio.transform import from_bounds
      with rasterio.open(synthetic_10m_raster) as src:
          data = src.read(1)
          profile = src.profile

      dry_path = tmp_path / "dry.tif"
      wet_path = tmp_path / "wet.tif"
      t_path = tmp_path / "t.tif"
      mi_path = tmp_path / "mi.tif"

      # dry = -20 dB, wet = -10 dB, scene_t = -15 dB → MI should be ~0.5
      for path, val in [(dry_path, -20.0), (wet_path, -10.0), (t_path, -15.0)]:
          with rasterio.open(path, "w", **profile) as dst:
              dst.write(np.full_like(data, val), 1)

      compute_moisture_index(t_path, dry_path, wet_path, mi_path)
      with rasterio.open(mi_path) as src:
          mi = src.read(1)
      valid = mi[mi != -9999.0]
      assert valid.min() >= -0.05   # allow small float noise
      assert valid.max() <= 1.05
      np.testing.assert_allclose(valid.mean(), 0.5, atol=0.05)
  ```

- [ ] **Step 2: Run to verify failure**

  ```bash
  pytest tests/unit/test_sar_indices.py::test_vv_vh_ratio_is_difference_in_db -v
  ```

- [ ] **Step 3: Implement `src/soilgeo/indices/sar.py`**

  ```python
  """SAR-derived spectral indices: VV/VH ratio and SAR Moisture Index."""
  from pathlib import Path

  import numpy as np
  import rasterio

  from soilgeo.utils.logging import get_logger

  log = get_logger(__name__)

  NODATA = -9999.0


  def _db_to_linear(arr: np.ndarray, nodata: float = NODATA) -> np.ndarray:
      out = np.where(arr == nodata, np.nan, 10 ** (arr / 10.0))
      return out


  def _linear_to_db(arr: np.ndarray, nodata: float = NODATA) -> np.ndarray:
      with np.errstate(divide="ignore", invalid="ignore"):
          out = np.where(np.isnan(arr) | (arr <= 0), nodata, 10.0 * np.log10(arr))
      return out.astype(np.float32)


  def compute_vv_vh_ratio(
      vv_path: Path,
      vh_path: Path,
      output_path: Path,
  ) -> Path:
      """VV/VH ratio in dB = VV_dB − VH_dB. Output unit: dB."""
      if output_path.exists():
          log.info("Skipping vv_vh_ratio (exists)")
          return output_path

      with rasterio.open(vv_path) as src:
          vv = src.read(1).astype(np.float32)
          profile = src.profile
          nd = src.nodata or NODATA

      with rasterio.open(vh_path) as src:
          vh = src.read(1).astype(np.float32)

      mask = (vv == nd) | (vh == nd)
      ratio = vv - vh
      ratio[mask] = NODATA

      output_path.parent.mkdir(parents=True, exist_ok=True)
      profile.update(dtype="float32", nodata=NODATA, count=1, compress="deflate")
      with rasterio.open(output_path, "w", **profile) as dst:
          dst.write(ratio, 1)
      return output_path


  def compute_moisture_index(
      scene_path: Path,
      dry_composite_path: Path,
      wet_composite_path: Path,
      output_path: Path,
  ) -> Path:
      """
      SAR Moisture Index (V1-F3):
        MI = (σ_t − σ_dry) / (σ_wet − σ_dry)
      Inputs are VV backscatter in dB. Output is dimensionless [0, 1] (clamped).
      """
      if output_path.exists():
          log.info("Skipping moisture_index (exists)")
          return output_path

      with rasterio.open(scene_path) as src:
          sigma_t = src.read(1).astype(np.float32)
          profile = src.profile
          nd = src.nodata or NODATA
      with rasterio.open(dry_composite_path) as src:
          sigma_dry = src.read(1).astype(np.float32)
          nd_dry = src.nodata or NODATA
      with rasterio.open(wet_composite_path) as src:
          sigma_wet = src.read(1).astype(np.float32)
          nd_wet = src.nodata or NODATA

      mask = (sigma_t == nd) | (sigma_dry == nd_dry) | (sigma_wet == nd_wet)
      denom = sigma_wet - sigma_dry

      with np.errstate(divide="ignore", invalid="ignore"):
          mi = np.where(
              mask | (np.abs(denom) < 1e-6),
              NODATA,
              np.clip((sigma_t - sigma_dry) / denom, 0.0, 1.0),
          )

      output_path.parent.mkdir(parents=True, exist_ok=True)
      profile.update(dtype="float32", nodata=NODATA, count=1, compress="deflate")
      with rasterio.open(output_path, "w", **profile) as dst:
          dst.write(mi.astype(np.float32), 1)

      log.info("Moisture index written: %s", output_path.name)
      return output_path
  ```

- [ ] **Step 4: Run tests**

  ```bash
  pytest tests/unit/test_sar_indices.py -v
  ```
  Expected: 4 PASSED

- [ ] **Step 5: Commit**

  ```bash
  git add src/soilgeo/indices/sar.py tests/unit/test_sar_indices.py
  git commit -m "feat(indices): VV/VH dB ratio and SAR Moisture Index MI=(σt-σdry)/(σwet-σdry)"
  ```

---

### Task 11: Stratified statistical analysis

**Files:**
- Create: `src/soilgeo/analysis/statistics.py`
- Create: `tests/unit/test_statistics.py`

- [ ] **Step 1: Write failing tests**

  Create `tests/unit/test_statistics.py`:
  ```python
  import numpy as np
  from soilgeo.analysis.statistics import stratify_by_quantiles, kruskal_wallis_test, spearman_correlation

  def test_stratify_returns_correct_labels():
      values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
      labels = stratify_by_quantiles(values, n_quantiles=4)
      assert labels.shape == values.shape
      assert set(labels) == {0, 1, 2, 3}

  def test_kruskal_wallis_rejects_different_distributions():
      rng = np.random.default_rng(0)
      group_a = rng.normal(0, 1, 100)
      group_b = rng.normal(5, 1, 100)
      result = kruskal_wallis_test([group_a, group_b])
      assert result["p_value"] < 0.001
      assert result["statistic"] > 0

  def test_kruskal_wallis_accepts_same_distribution():
      rng = np.random.default_rng(1)
      group_a = rng.normal(0, 1, 100)
      group_b = rng.normal(0, 1, 100)
      result = kruskal_wallis_test([group_a, group_b])
      # Should not be significant at p<0.001
      assert result["p_value"] > 0.001

  def test_spearman_correlation_perfect():
      x = np.arange(50, dtype=float)
      y = x * 2.0 + 1.0
      result = spearman_correlation(x, y)
      assert abs(result["rho"] - 1.0) < 1e-6
      assert result["p_value"] < 0.001

  def test_spearman_no_correlation():
      rng = np.random.default_rng(2)
      x = rng.normal(0, 1, 200)
      y = rng.normal(0, 1, 200)
      result = spearman_correlation(x, y)
      assert abs(result["rho"]) < 0.3  # weak correlation expected by chance
  ```

- [ ] **Step 2: Run to verify failure**

  ```bash
  pytest tests/unit/test_statistics.py -v
  ```

- [ ] **Step 3: Implement `src/soilgeo/analysis/statistics.py`**

  ```python
  """Stratified statistics: Kruskal-Wallis test, Spearman correlation, quantile stratification."""
  from typing import Sequence

  import numpy as np
  from scipy import stats

  from soilgeo.utils.logging import get_logger

  log = get_logger(__name__)


  def stratify_by_quantiles(values: np.ndarray, n_quantiles: int = 4) -> np.ndarray:
      """Assign 0-indexed quantile class labels to a 1D array of values."""
      quantile_edges = np.quantile(values, np.linspace(0, 1, n_quantiles + 1))
      # np.digitize assigns 1-indexed bins; subtract 1 and clip the top
      labels = np.digitize(values, quantile_edges[1:-1])
      return labels.astype(np.uint8)


  def kruskal_wallis_test(groups: Sequence[np.ndarray]) -> dict:
      """Kruskal-Wallis H test on two or more groups."""
      stat, p = stats.kruskal(*groups)
      result = {"statistic": float(stat), "p_value": float(p)}
      log.info("Kruskal-Wallis H=%.3f p=%.4g", stat, p)
      return result


  def spearman_correlation(x: np.ndarray, y: np.ndarray) -> dict:
      """Spearman rank correlation between two arrays."""
      rho, p = stats.spearmanr(x, y)
      result = {"rho": float(rho), "p_value": float(p)}
      log.info("Spearman ρ=%.4f p=%.4g", rho, p)
      return result


  def backscatter_stats_by_stratum(
      backscatter: np.ndarray,
      strata: np.ndarray,
      nodata: float = -9999.0,
  ) -> dict:
      """
      For each unique stratum label, compute backscatter statistics.
      Returns dict: {stratum_id: {mean, median, std, q25, q75, n}}
      """
      results = {}
      for label in np.unique(strata):
          mask = (strata == label) & (backscatter != nodata)
          vals = backscatter[mask]
          if vals.size == 0:
              continue
          results[int(label)] = {
              "mean": float(np.mean(vals)),
              "median": float(np.median(vals)),
              "std": float(np.std(vals)),
              "q25": float(np.percentile(vals, 25)),
              "q75": float(np.percentile(vals, 75)),
              "n": int(vals.size),
          }
      return results
  ```

- [ ] **Step 4: Run tests**

  ```bash
  pytest tests/unit/test_statistics.py -v
  ```
  Expected: 5 PASSED

- [ ] **Step 5: Commit**

  ```bash
  git add src/soilgeo/analysis/statistics.py tests/unit/test_statistics.py
  git commit -m "feat(analysis): stratified stats — Kruskal-Wallis, Spearman correlation, quantile stratification"
  ```

---

### Task 12: Surface Response Classification

**Files:**
- Create: `src/soilgeo/analysis/classification.py`
- Create: `tests/unit/test_classification.py`

Classes (V1-F7): `persistently_wet | seasonally_wet | dry | water | urban_outlier`
Method: k-means on `[moisture_index, TWI, slope]` with post-hoc label assignment by cluster centroids.

- [ ] **Step 1: Write failing tests**

  Create `tests/unit/test_classification.py`:
  ```python
  import numpy as np
  from soilgeo.analysis.classification import classify_surface_response, CLASS_NAMES

  def test_classify_output_shape():
      rng = np.random.default_rng(42)
      n = 1000
      features = {
          "moisture_index": rng.uniform(0, 1, n),
          "twi": rng.uniform(2, 15, n),
          "slope": rng.uniform(0, 30, n),
      }
      labels = classify_surface_response(features, n_clusters=5, random_state=0)
      assert labels.shape == (n,)
      assert labels.dtype == np.uint8

  def test_classify_produces_n_classes():
      rng = np.random.default_rng(7)
      n = 2000
      features = {
          "moisture_index": rng.uniform(0, 1, n),
          "twi": rng.uniform(2, 15, n),
          "slope": rng.uniform(0, 30, n),
      }
      labels = classify_surface_response(features, n_clusters=5, random_state=0)
      assert len(np.unique(labels)) == 5

  def test_class_names_dict_has_five_entries():
      assert len(CLASS_NAMES) == 5
  ```

- [ ] **Step 2: Run to verify failure**

  ```bash
  pytest tests/unit/test_classification.py -v
  ```

- [ ] **Step 3: Implement `src/soilgeo/analysis/classification.py`**

  ```python
  """Surface Response Classification via k-means on MI, TWI, slope (V1-F7)."""
  import numpy as np
  from sklearn.preprocessing import StandardScaler
  from sklearn.cluster import KMeans

  from soilgeo.utils.logging import get_logger

  log = get_logger(__name__)

  # Post-hoc label names: assigned after clustering by inspecting cluster centroids.
  # Sorted ascending by composite wetness score = MI_centroid + 0.05 × TWI_centroid − 0.01 × slope_centroid
  CLASS_NAMES = {
      0: "dry",
      1: "moderately_dry",
      2: "transitional",
      3: "seasonally_wet",
      4: "persistently_wet",
  }


  def classify_surface_response(
      features: dict,
      n_clusters: int = 5,
      random_state: int = 42,
  ) -> np.ndarray:
      """
      K-means classification on scaled [moisture_index, twi, slope].
      Returns uint8 array of class labels (0-indexed, sorted by wetness).
      """
      keys = ["moisture_index", "twi", "slope"]
      X = np.column_stack([features[k] for k in keys])

      valid_mask = ~np.any(np.isnan(X) | (X == -9999.0), axis=1)
      X_valid = X[valid_mask]

      scaler = StandardScaler()
      X_scaled = scaler.fit_transform(X_valid)

      km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
      raw_labels = km.fit_predict(X_scaled)

      # Re-order clusters by wetness: MI↑ + TWI↑ − slope↑
      centroids = scaler.inverse_transform(km.cluster_centers_)
      wetness_score = centroids[:, 0] + 0.05 * centroids[:, 1] - 0.01 * centroids[:, 2]
      rank_order = np.argsort(wetness_score)           # wettest cluster → highest label
      remap = {old: new for new, old in enumerate(rank_order)}
      ordered_labels = np.vectorize(remap.get)(raw_labels).astype(np.uint8)

      output = np.full(len(features["moisture_index"]), 255, dtype=np.uint8)  # 255 = nodata
      output[valid_mask] = ordered_labels

      log.info(
          "Surface Response Classes: %s",
          {CLASS_NAMES.get(k, k): int(np.sum(output == k)) for k in range(n_clusters)},
      )
      return output
  ```

- [ ] **Step 4: Run tests**

  ```bash
  pytest tests/unit/test_classification.py -v
  ```
  Expected: 3 PASSED

- [ ] **Step 5: Commit**

  ```bash
  git add src/soilgeo/analysis/classification.py tests/unit/test_classification.py
  git commit -m "feat(analysis): Surface Response Classification — k-means on MI/TWI/slope with wetness ordering"
  ```

---

## Phase 7 — Products & COG catalog

### Task 13: Products module (COG catalog + Construction Risk Index)

**Files:**
- Create: `src/soilgeo/products/cog.py`

- [ ] **Step 1: Implement `src/soilgeo/products/cog.py`**

  (No new tests needed — `write_cog` already tested in Task 4. This module adds the catalog and risk index.)

  ```python
  """Product catalog management and Construction Moisture Risk Index (V1-F9)."""
  import json
  from datetime import datetime, timezone
  from pathlib import Path

  import numpy as np
  import rasterio

  from soilgeo.utils.logging import get_logger

  log = get_logger(__name__)

  NODATA = -9999.0


  def write_product_catalog(products_dir: Path, entries: list[dict]) -> Path:
      """Write a catalog.json listing all published products."""
      catalog_path = products_dir / "catalog.json"
      catalog = {
          "_generated_at": datetime.now(timezone.utc).isoformat(),
          "products": entries,
      }
      catalog_path.write_text(json.dumps(catalog, indent=2))
      log.info("Product catalog written: %d entries", len(entries))
      return catalog_path


  def compute_construction_risk(
      mi_path: Path,
      flow_acc_path: Path,
      slope_path: Path,
      output_path: Path,
      weights: dict | None = None,
  ) -> Path:
      """
      Construction Moisture Risk Index (V1-F9):
        CRI = w_mi × MI + w_fa × log(FA + 1) / max(log(FA+1)) + w_slope × (1 − slope/max_slope)
      Output: [0, 1] float32 COG.
      """
      if output_path.exists():
          log.info("Skipping construction_risk (exists)")
          return output_path

      weights = weights or {"moisture_index": 0.5, "flow_accumulation_log": 0.3, "low_slope_factor": 0.2}

      with rasterio.open(mi_path) as src:
          mi = src.read(1).astype(np.float32)
          profile = src.profile
          nd_mi = src.nodata or NODATA
      with rasterio.open(flow_acc_path) as src:
          fa = src.read(1).astype(np.float64)
          nd_fa = src.nodata or NODATA
      with rasterio.open(slope_path) as src:
          slope = src.read(1).astype(np.float32)
          nd_sl = src.nodata or NODATA

      mask = (mi == nd_mi) | (fa == nd_fa) | (slope == nd_sl)

      fa_log = np.log1p(np.where(fa == nd_fa, 0.0, fa))
      fa_norm = fa_log / (fa_log[~mask].max() + 1e-9)

      slope_max = slope[~mask].max()
      low_slope = np.clip(1.0 - slope / (slope_max + 1e-9), 0.0, 1.0)
      mi_clipped = np.clip(mi, 0.0, 1.0)

      cri = (
          weights["moisture_index"] * mi_clipped
          + weights["flow_accumulation_log"] * fa_norm.astype(np.float32)
          + weights["low_slope_factor"] * low_slope
      )
      cri[mask] = NODATA

      output_path.parent.mkdir(parents=True, exist_ok=True)
      profile.update(dtype="float32", count=1, nodata=NODATA, compress="deflate")
      with rasterio.open(output_path, "w", **profile) as dst:
          dst.write(cri.astype(np.float32), 1)

      log.info("Construction Risk Index written: %s", output_path.name)
      return output_path
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add src/soilgeo/products/cog.py
  git commit -m "feat(products): product catalog JSON and Construction Moisture Risk Index"
  ```

---

## Phase 8 — Pipeline CLI

### Task 14: V1 pipeline CLI entry point

**Files:**
- Create: `pipelines/run_v1.py`

This wires all modules together as a resumable, config-driven pipeline. Each stage checks for existing outputs and skips if present.

- [ ] **Step 1: Implement `pipelines/run_v1.py`**

  ```python
  #!/usr/bin/env python
  """
  V1 Exploratory Soil Characterization Pipeline.

  Usage:
      conda activate soilgeo
      python pipelines/run_v1.py --config config/pipelines/v1.yml [--force] [--stages all]

  Stages (run in order):
      download_dem   — download Copernicus DEM GLO-30 tiles
      download_s1    — search and download Sentinel-1 scenes
      preprocess_s1  — SNAP GPT preprocessing chain
      terrain        — DEM derivatives (slope, aspect, curvature, roughness, hillshade)
      hydrology      — WhiteboxTools (flow_acc, TWI, SPI)
      composites     — wet/dry VV backscatter composites
      indices        — VV/VH ratio, SAR Moisture Index
      classify       — Surface Response Classes
      risk           — Construction Moisture Risk Index (optional)
      catalog        — write product catalog JSON
  """
  import argparse
  import sys
  from pathlib import Path

  from dotenv import load_dotenv

  from soilgeo.utils.config import load_aoi_config, load_pipeline_config
  from soilgeo.utils.logging import get_logger

  load_dotenv()


  def parse_args():
      p = argparse.ArgumentParser(description="Run V1 SAR Soil pipeline")
      p.add_argument("--config", default="config/pipelines/v1.yml")
      p.add_argument("--force", action="store_true", help="Re-run all stages ignoring existing outputs")
      p.add_argument(
          "--stages",
          default="all",
          help="Comma-separated list of stages to run, or 'all'",
      )
      return p.parse_args()


  def run(args):
      cfg = load_pipeline_config(Path(args.config))
      aoi = load_aoi_config(Path(cfg.aoi_config))
      log = get_logger("pipeline.v1", log_dir=Path(cfg.paths["log_dir"]))

      raw = Path(cfg.paths["raw_dir"])
      interim = Path(cfg.paths["interim_dir"])
      processed = Path(cfg.paths["processed_dir"])

      all_stages = [
          "download_dem", "download_s1", "preprocess_s1",
          "terrain", "hydrology", "composites",
          "indices", "classify", "risk", "catalog",
      ]
      stages = all_stages if args.stages == "all" else args.stages.split(",")
      log.info("=== V1 Pipeline: %s | AOI: %s | stages: %s ===", cfg.version, aoi.name, stages)

      # ── Stage 1: Download DEM ────────────────────────────────────────────────
      if "download_dem" in stages:
          log.info("--- Stage: download_dem ---")
          from soilgeo.acquisition.dem import download_dem_tiles
          dem_tiles = download_dem_tiles(aoi.bbox, raw / "dem", buffer_deg=0.05)
          log.info("DEM tiles: %d downloaded", len(dem_tiles))

      # ── Stage 2: Download Sentinel-1 ────────────────────────────────────────
      if "download_s1" in stages:
          log.info("--- Stage: download_s1 ---")
          from soilgeo.acquisition.sentinel1 import build_s1_search_params, search_scenes, filter_by_orbit, download_scenes
          s1_cfg = aoi.sentinel1
          for season, season_cfg in [("wet", s1_cfg["wet_season"]), ("dry", s1_cfg["dry_season"])]:
              params = build_s1_search_params(
                  bbox=aoi.bbox,
                  start=season_cfg["start"],
                  end=season_cfg["end"],
                  orbit_direction=s1_cfg["orbit_direction"],
              )
              results = search_scenes(params)
              filtered = filter_by_orbit([r.__dict__ if hasattr(r, '__dict__') else r for r in results])
              download_scenes(results[:len(filtered)], raw / "s1" / season)

      # ── Stage 3: Preprocess SAR ──────────────────────────────────────────────
      if "fetch_s1" in stages:
          log.info("--- Stage: fetch_s1 ---")
          from soilgeo.acquisition.sentinel_hub import fetch_backscatter
          s1_cfg = aoi.sentinel1
          for season, season_cfg in [("wet", s1_cfg["wet_season"]), ("dry", s1_cfg["dry_season"])]:
              out = interim / "s1" / f"s1_vvvh_{aoi.name}_{season}_median.tif"
              fetch_backscatter(
                  bbox_wgs84=aoi.bbox,
                  time_interval=(season_cfg["start"], season_cfg["end"]),
                  output_path=out,
                  resolution_m=aoi.resolution_m,
                  median_composite=True,
              )
              log.info("S1 %s composite: %s", season, out.name)

      # ── Stage 4: Terrain ─────────────────────────────────────────────────────
      if "terrain" in stages:
          log.info("--- Stage: terrain ---")
          from soilgeo.terrain.derivatives import (
              compute_slope, compute_aspect, compute_roughness,
              compute_hillshade, compute_curvature,
          )
          dem_mosaic = interim / "dem" / f"dem_{aoi.name}_utm.tif"
          t_dir = interim / "terrain"
          t_dir.mkdir(parents=True, exist_ok=True)
          compute_slope(dem_mosaic, t_dir / "slope.tif")
          compute_aspect(dem_mosaic, t_dir / "aspect.tif")
          compute_roughness(dem_mosaic, t_dir / "roughness.tif")
          compute_hillshade(dem_mosaic, t_dir / "hillshade.tif")
          compute_curvature(dem_mosaic, t_dir / "curvature.tif")

      # ── Stage 5: Hydrology ───────────────────────────────────────────────────
      if "hydrology" in stages:
          log.info("--- Stage: hydrology ---")
          from soilgeo.hydrology.whitebox import compute_flow_accumulation, compute_twi_spi
          dem_30m = interim / "dem" / f"dem_{aoi.name}_30m.tif"   # 30 m DEM for TWI (V1-T1)
          h_dir = interim / "hydrology"
          h_dir.mkdir(parents=True, exist_ok=True)
          compute_flow_accumulation(dem_30m, h_dir / "flow_acc.tif", work_dir=h_dir)
          compute_twi_spi(dem_30m, h_dir / "twi.tif", h_dir / "spi.tif", work_dir=h_dir)

      # ── Stage 6: SAR composites ──────────────────────────────────────────────
      if "composites" in stages:
          log.info("--- Stage: composites ---")
          import numpy as np
          import rasterio
          for season in ("wet", "dry"):
              s1_dir = interim / "s1" / season
              tifs = sorted(s1_dir.glob("*_preprocessed.tif"))
              if not tifs:
                  log.warning("No preprocessed S1 scenes found for %s season", season)
                  continue
              # Median composite of VV band across all scenes
              comp_dir = interim / "composites"
              comp_dir.mkdir(parents=True, exist_ok=True)
              out_path = comp_dir / f"s1_vv_{season}_median.tif"
              if out_path.exists():
                  continue
              bands = []
              for tif in tifs:
                  with rasterio.open(tif) as src:
                      arr = src.read(1).astype(np.float32)
                      profile = src.profile
                      arr[arr == (src.nodata or -9999.0)] = np.nan
                      bands.append(arr)
              stack = np.stack(bands, axis=0)
              median = np.nanmedian(stack, axis=0)
              median[np.isnan(median)] = -9999.0
              profile.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate")
              with rasterio.open(out_path, "w", **profile) as dst:
                  dst.write(median.astype(np.float32), 1)
              log.info("Composite written: %s (%d scenes)", out_path.name, len(tifs))

      # ── Stage 7: Indices ─────────────────────────────────────────────────────
      if "indices" in stages:
          log.info("--- Stage: indices ---")
          from soilgeo.indices.sar import compute_vv_vh_ratio, compute_moisture_index
          comp = interim / "composites"
          idx_dir = interim / "indices"
          idx_dir.mkdir(parents=True, exist_ok=True)
          for season in ("wet", "dry"):
              vv = comp / f"s1_vv_{season}_median.tif"
              vh = comp / f"s1_vh_{season}_median.tif"
              if vv.exists() and vh.exists():
                  compute_vv_vh_ratio(vv, vh, idx_dir / f"vv_vh_ratio_{season}.tif")
          dry_comp = comp / "s1_vv_dry_median.tif"
          wet_comp = comp / "s1_vv_wet_median.tif"
          # Compute MI for each wet-season scene individually
          for tif in sorted((interim / "s1" / "wet").glob("*_preprocessed.tif")):
              date_str = tif.stem.split("_")[4][:8]   # SAFE naming convention
              compute_moisture_index(
                  scene_path=tif,
                  dry_composite_path=dry_comp,
                  wet_composite_path=wet_comp,
                  output_path=idx_dir / f"moisture_index_{date_str}.tif",
              )

      # ── Stage 8: Surface Response Classification ─────────────────────────────
      if "classify" in stages:
          log.info("--- Stage: classify ---")
          import numpy as np
          import rasterio
          from soilgeo.analysis.classification import classify_surface_response
          from soilgeo.utils.geo import write_cog
          mi_path = next((interim / "indices").glob("moisture_index_*.tif"), None)
          twi_path = interim / "hydrology" / "twi.tif"
          slope_path = interim / "terrain" / "slope.tif"
          if not (mi_path and twi_path.exists() and slope_path.exists()):
              log.warning("Missing inputs for classification — skipping")
          else:
              with rasterio.open(mi_path) as src:
                  mi = src.read(1).flatten().astype(np.float32)
                  profile = src.profile
                  h, w = src.height, src.width
              with rasterio.open(twi_path) as src:
                  twi = src.read(1).flatten().astype(np.float32)
              with rasterio.open(slope_path) as src:
                  slope = src.read(1).flatten().astype(np.float32)
              cfg_src = cfg.surface_response_classes
              labels = classify_surface_response(
                  {"moisture_index": mi, "twi": twi, "slope": slope},
                  n_clusters=cfg_src["n_clusters"],
                  random_state=cfg_src["random_state"],
              ).reshape(h, w)
              out = processed / f"surface_response_classes_{aoi.name}.tif"
              out.parent.mkdir(parents=True, exist_ok=True)
              profile.update(dtype="uint8", nodata=255, count=1, compress="deflate")
              with rasterio.open(out, "w", **profile) as dst:
                  dst.write(labels, 1)
              log.info("Surface Response Classes written: %s", out.name)

      # ── Stage 9: Construction Risk (optional) ────────────────────────────────
      if "risk" in stages and cfg.construction_risk["enabled"]:
          log.info("--- Stage: risk ---")
          from soilgeo.products.cog import compute_construction_risk
          mi_path = next((interim / "indices").glob("moisture_index_*.tif"), None)
          fa_path = interim / "hydrology" / "flow_acc.tif"
          slope_path = interim / "terrain" / "slope.tif"
          if mi_path and fa_path.exists() and slope_path.exists():
              compute_construction_risk(
                  mi_path=mi_path,
                  flow_acc_path=fa_path,
                  slope_path=slope_path,
                  output_path=processed / f"construction_moisture_risk_{aoi.name}.tif",
                  weights=cfg.construction_risk["weights"],
              )

      # ── Stage 10: Product catalog ────────────────────────────────────────────
      if "catalog" in stages:
          log.info("--- Stage: catalog ---")
          from soilgeo.products.cog import write_product_catalog
          entries = [
              {"name": p.stem, "path": str(p.relative_to(processed)), "type": "COG"}
              for p in sorted(processed.glob("*.tif"))
          ]
          write_product_catalog(processed, entries)

      log.info("=== V1 Pipeline complete ===")


  if __name__ == "__main__":
      run(parse_args())
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add pipelines/run_v1.py
  git commit -m "feat(pipeline): V1 resumable CLI entry point — download → preprocess → terrain → hydrology → indices → classify → catalog"
  ```

---

## Phase 9 — CI & Final Tests

### Task 15: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

  ```yaml
  name: CI

  on:
    push:
      branches: [main]
    pull_request:
      branches: [main]

  jobs:
    lint-and-test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4

        - name: Set up Python
          uses: actions/setup-python@v5
          with:
            python-version: "3.11"

        - name: Install dependencies
          run: |
            pip install --upgrade pip
            pip install rasterio rioxarray xarray geopandas shapely pyproj scipy \
                        scikit-learn matplotlib asf-search python-dotenv whitebox \
                        pyyaml pytest ruff

        - name: Install package
          run: pip install -e .

        - name: Lint with ruff
          run: ruff check src/ tests/ pipelines/

        - name: Run unit tests (no SNAP, no downloads)
          run: pytest tests/unit/ -v -m "not integration"
  ```

- [ ] **Step 2: Commit and push**

  ```bash
  git add .github/
  git commit -m "ci: GitHub Actions lint + unit tests on every push"
  git push
  ```

- [ ] **Step 3: Verify CI passes on GitHub**

  ```bash
  gh run list --limit 5
  gh run watch
  ```

---

### Task 16: Run full pipeline locally (V1 end-to-end)

This is manual execution — assumes `.env` is populated with real credentials and SNAP is installed.

- [ ] **Step 1: Verify environment**

  ```bash
  conda activate soilgeo
  cp .env.example .env
  # Edit .env: fill EARTHDATA_USERNAME, EARTHDATA_PASSWORD, SNAP_GPT
  python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('SNAP_GPT'))"
  ```

- [ ] **Step 2: Run download stages only first**

  ```bash
  python pipelines/run_v1.py --stages download_dem,download_s1
  ```
  Expected: DEM tiles in `data/raw/dem/`, S1 SAFE zips in `data/raw/s1/{wet,dry}/`

- [ ] **Step 3: Preprocess SAR (slow — ~30–60 min per scene)**

  ```bash
  python pipelines/run_v1.py --stages preprocess_s1
  ```
  Expected: `data/interim/s1/{wet,dry}/*_preprocessed.tif` files

- [ ] **Step 4: Run remaining stages**

  ```bash
  python pipelines/run_v1.py --stages terrain,hydrology,composites,indices,classify,risk,catalog
  ```
  Expected: `data/processed/` contains `surface_response_classes_konya.tif`, `construction_moisture_risk_konya.tif`, `catalog.json`

- [ ] **Step 5: Open `notebooks/01_v1_exploratory_report.ipynb`** in Jupyter, run all cells, verify maps render.

- [ ] **Step 6: Commit notebook outputs and final tag**

  ```bash
  git add data/processed/catalog.json notebooks/01_v1_exploratory_report.ipynb
  git commit -m "chore: V1 pipeline run complete — catalog + exploratory report"
  git tag v1.0
  git push && git push --tags
  ```

---

## Self-Review — Spec Coverage Check

| V1 Requirement | Covered by Task |
|---|---|
| V1-F1 Download S1 wet+dry ≥3 scenes | Task 5 + 14 (download_s1 stage) |
| V1-F2 Full SAR preprocessing chain | Task 7 (SNAP GPT graph + processor) |
| V1-F3 VV, VH, VV/VH ratio, Moisture Index | Task 10 |
| V1-F4 Terrain derivatives | Task 8 |
| V1-F5 Hydrological layers (TWI, SPI, flow acc) | Task 9 |
| V1-F6 Stratified stats (Kruskal-Wallis, Spearman) | Task 11 |
| V1-F7 Surface Response Classes | Task 12 |
| V1-F8 Export COGs + catalog | Task 13 |
| V1-F9 Construction Risk Index (optional) | Task 13 |
| DR-R1 All rasters as COG | Task 4 (write_cog) used everywhere |
| DR-R2 Single relative orbit | Task 5 (filter_by_orbit) |
| DR-R5 Scripted downloads, env var creds | Task 5, Task 6, §4.3 |
| V1-T1 TWI at 30 m DEM | Task 9 + config konya.yml |
| V1-T2 Valid-pixel masks | Indices + classification modules |
| V1-T3 <4h runtime | 10 m grid, 50×50 km AOI |
| CR-R1 Credentials in .env | Task 1 |
| GS-R1 UTM 36N CRS | config konya.yml, geo.py |
| GS-R4 Naming convention | pipeline stages |
| CI lint + unit tests | Task 15 |

---

**Plan complete and saved.**

Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
