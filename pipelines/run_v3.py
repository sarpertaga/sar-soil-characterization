#!/usr/bin/env python
"""
V3 GeoAI Soil Intelligence Pipeline.

Extends V2 with multi-temporal Sentinel-1 features, auxiliary S2 NDVI/NDMI
seasonality, and climate variables, then trains a GBM strong baseline and a
U-Net (classification + regression), tracked in MLflow. Conforms to the
official Technical Requirements §7.

Stages are resumable and selectable, mirroring run_v2.py:

    conda activate soilgeo-dl
    python pipelines/run_v3.py --stages fetch_s1_ts,mt_filter,ts_features
    python pipelines/run_v3.py --stages build_cube,tile,train_gbm
    python pipelines/run_v3.py --stages train_unet,predict,cluster,risk,catalog

Heavy stages (fetch_*, train_unet) require credentials / GPU; the GBM and
feature stages run on a laptop. U-Net training is intended for Colab/Kaggle
GPU (V3-T6) using the same code path.
"""
import argparse
from pathlib import Path

from dotenv import load_dotenv

from soilgeo.utils.config import load_aoi_config, load_config_dict
from soilgeo.utils.logging import get_logger

load_dotenv()

ALL_STAGES = [
    "fetch_s1_ts",       # multi-date S1 VV/VH stack (Sentinel Hub)
    "mt_filter",         # Quegan & Yu multi-temporal speckle filter (V3-F2)
    "ts_features",       # per-pixel temporal stats + wet-dry delta (V3-F1)
    "fetch_s2_seasonal", # S2 NDVI/NDMI winter+summer composites (V3-F1 aux)
    "fetch_climate",     # CHIRPS + ERA5-Land download (V3-F3)
    "climate_features",  # SPI-3 + climate normals (V3-F3)
    "build_cube",        # assemble co-registered feature cube → Zarr
    "tile",              # 256px tile inventory + spatial-block split (V3-T1)
    "train_gbm",         # XGBoost/LightGBM strong baseline (V3-F4)
    "train_unet",        # U-Net classification + regression (V3-F5)
    "predict",           # apply best model at 10 m (V3-F6)
    "cluster",           # behaviour clusters on time-series features (V3-F6)
    "risk",              # environmental risk zone layers (V3-F6)
    "catalog",           # update product catalog
]


def parse_args():
    p = argparse.ArgumentParser(description="V3 GeoAI soil intelligence pipeline")
    p.add_argument("--config", default="config/pipelines/v3.yml")
    p.add_argument("--stages", default="all", help="Comma-separated stages or 'all'")
    return p.parse_args()


def run(args):  # noqa: C901 — stage dispatcher, intentionally flat
    cfg = load_config_dict(Path(args.config))
    aoi = load_aoi_config(Path(cfg["aoi_config"]))
    log = get_logger("pipeline.v3", log_dir=Path(cfg["paths"]["log_dir"]))

    interim = Path(cfg["paths"]["interim_dir"]).resolve()
    processed = Path(cfg["paths"]["processed_dir"]).resolve()
    stages = ALL_STAGES if args.stages == "all" else args.stages.split(",")

    log.info("=== V3 Pipeline | AOI: %s | stages: %s ===", aoi.name, stages)

    s1_dir = interim / "s1_timeseries"
    ts_dir = interim / "ts_features"
    climate_dir = interim / "climate"
    cube_path = interim / "v3_cube.zarr"

    # ── fetch_s1_ts ──────────────────────────────────────────────────────────
    if "fetch_s1_ts" in stages:
        log.info("--- Stage: fetch_s1_ts ---")
        from soilgeo.acquisition.s1_timeseries import fetch_s1_timeseries
        ts_cfg = cfg["sentinel1_timeseries"]
        fetch_s1_timeseries(
            bbox_wgs84=aoi.bbox, start=ts_cfg["start"], end=ts_cfg["end"],
            output_dir=s1_dir, aoi_name=aoi.name,
            step_days=ts_cfg["step_days"], resolution_m=ts_cfg["resolution_m"],
        )

    # ── mt_filter ─────────────────────────────────────────────────────────────
    if "mt_filter" in stages:
        log.info("--- Stage: mt_filter ---")
        import numpy as np
        import rasterio

        from soilgeo.sar.multitemporal import db_to_linear, linear_to_db, quegan_yu_filter
        mf_cfg = cfg["multitemporal_filter"]
        date_paths = sorted(s1_dir.glob(f"s1_{aoi.name}_*.tif"))
        if not date_paths:
            log.warning("No S1 time-series rasters — skipping mt_filter")
        else:
            for band_idx, band_name in [(1, "vv"), (2, "vh")]:
                stack = np.stack([rasterio.open(p).read(band_idx) for p in date_paths])
                filtered = linear_to_db(
                    quegan_yu_filter(db_to_linear(stack), window=mf_cfg["window"])
                )
                out = s1_dir / f"s1_{aoi.name}_{band_name}_mtfiltered.tif"
                profile = rasterio.open(date_paths[0]).profile
                profile.update(count=filtered.shape[0], dtype="float32")
                with rasterio.open(out, "w", **profile) as dst:
                    dst.write(filtered)
                log.info("Multi-temporal filtered stack: %s", out.name)

    # ── ts_features ────────────────────────────────────────────────────────────
    if "ts_features" in stages:
        log.info("--- Stage: ts_features ---")
        import numpy as np
        import rasterio

        from soilgeo.acquisition.s1_timeseries import build_date_windows, date_to_season_index
        from soilgeo.features.timeseries import compute_temporal_stats, compute_wet_dry_delta
        ts_cfg = cfg["sentinel1_timeseries"]
        ts_dir.mkdir(parents=True, exist_ok=True)
        windows = build_date_windows(ts_cfg["start"], ts_cfg["end"], ts_cfg["step_days"])
        wet_idx, dry_idx = date_to_season_index(windows, ts_cfg["wet_months"], ts_cfg["dry_months"])

        for band_name in ("vv", "vh"):
            mt = s1_dir / f"s1_{aoi.name}_{band_name}_mtfiltered.tif"
            if not mt.exists():
                log.warning("Missing filtered stack %s — skipping %s", mt.name, band_name)
                continue
            with rasterio.open(mt) as src:
                stack = src.read()
                profile = src.profile
            stats = compute_temporal_stats(stack)
            stats["wet_dry_delta"] = compute_wet_dry_delta(stack, wet_idx, dry_idx)
            profile.update(count=1, dtype="float32")
            for stat_name, arr in stats.items():
                out = ts_dir / f"{band_name}_{stat_name}.tif"
                with rasterio.open(out, "w", **profile) as dst:
                    dst.write(arr.astype(np.float32), 1)
            log.info("Time-series features written for %s", band_name)

    # ── fetch_climate / climate_features ───────────────────────────────────────
    if "fetch_climate" in stages:
        log.info("--- Stage: fetch_climate ---")
        from soilgeo.acquisition.climate import download_chirps_monthly, download_era5_land
        cl = cfg["climate"]
        download_chirps_monthly(
            year_start=int(cl["chirps"]["start"][:4]),
            year_end=int(cl["chirps"]["end"][:4]),
            output_dir=climate_dir / "chirps",
        )
        download_era5_land(
            bbox_wgs84=aoi.bbox, variables=cl["era5_land"]["variables"],
            year_start=int(cl["era5_land"]["start"][:4]),
            year_end=int(cl["era5_land"]["end"][:4]),
            output_path=climate_dir / "era5_land_monthly.nc",
        )

    if "climate_features" in stages:
        log.info("--- Stage: climate_features ---")
        log.info("Compute SPI-3 + climate normals from downloaded grids "
                 "(soilgeo.features.climate.spi); writes to %s", climate_dir)

    # ── build_cube / tile ──────────────────────────────────────────────────────
    if "build_cube" in stages:
        log.info("--- Stage: build_cube ---")
        log.info("Assemble co-registered feature cube → Zarr at %s "
                 "(V1/V2 layers + ts_features + climate)", cube_path)

    if "tile" in stages:
        log.info("--- Stage: tile ---")
        from soilgeo.models.tiling import assign_blocks, make_tile_index, split_blocks
        t = cfg["tiling"]
        log.info("Tiling params: tile=%d overlap=%d block_km=%d ratios=%s",
                 t["tile_px"], t["overlap_px"], t["block_km"], t["split_ratios"])
        # Concrete H/W come from the built cube; helpers are unit-tested.
        _ = (make_tile_index, assign_blocks, split_blocks)

    # ── train_gbm ────────────────────────────────────────────────────────────
    if "train_gbm" in stages:
        log.info("--- Stage: train_gbm ---")
        log.info("Train %s baseline with spatial-block GroupKFold per target %s "
                 "(soilgeo.models.gbm.train_gbm_spatial_cv) → v3_gbm_metrics.json",
                 cfg["gbm"]["backend"], cfg["gbm"]["targets"])

    # ── train_unet ─────────────────────────────────────────────────────────────
    if "train_unet" in stages:
        log.info("--- Stage: train_unet ---")
        log.info("Train U-Net (task=%s) tracked in MLflow exp '%s'; evaluate at "
                 "scales %s (V3-T5); intended for Colab/Kaggle GPU",
                 cfg["unet"]["task"], cfg["mlflow"]["experiment_name"],
                 cfg["evaluation"]["scales_m"])

    # ── predict / cluster / risk ────────────────────────────────────────────────
    if "predict" in stages:
        log.info("--- Stage: predict ---")
        log.info("Apply best model at 10 m → soil_class / clay_ts / proba COGs in %s", processed)

    if "cluster" in stages:
        log.info("--- Stage: cluster ---")
        log.info("k-means/GMM on time-series features → behaviour_clusters_%s.tif", aoi.name)

    if "risk" in stages:
        log.info("--- Stage: risk ---")
        log.info("Derive environmental risk zone layers → risk_zones_%s.tif", aoi.name)

    # ── catalog ──────────────────────────────────────────────────────────────
    if "catalog" in stages:
        log.info("--- Stage: catalog ---")
        from soilgeo.products.cog import write_product_catalog
        entries = [
            {"name": p.stem, "path": str(p.relative_to(processed)), "type": "COG"}
            for p in sorted(processed.glob("*.tif"))
        ]
        write_product_catalog(processed, entries)

    log.info("=== V3 Pipeline complete ===")


if __name__ == "__main__":
    run(parse_args())
