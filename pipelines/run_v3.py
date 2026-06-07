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

NODATA = -9999.0

ALL_STAGES = [
    "fetch_s1_ts",       # multi-date S1 VV/VH stack (Sentinel Hub)
    "mt_filter",         # Quegan & Yu multi-temporal speckle filter (V3-F2)
    "ts_features",       # per-pixel temporal stats + wet-dry delta (V3-F1)
    "terrain",           # DEM → slope, curvature, TWI (static V1 layers)
    "fetch_s2_seasonal", # S2 NDVI/NDMI winter+summer composites (V3-F1 aux)
    "soilgrids",         # clay/sand/soc labels (V2 reuse)
    "fetch_climate",     # CHIRPS + ERA5-Land download (V3-F3)
    "climate_features",  # SPI-3 + climate normals (V3-F3)
    "build_cube",        # assemble feature cube → Zarr + matrix + spatial-block groups
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
    s2_dir = interim / "s2_seasonal"
    terrain_dir = interim / "terrain"
    hydro_dir = interim / "hydrology"
    climate_dir = interim / "climate"
    sg_dir = interim / "soilgrids"
    dem_path = interim / "dem" / f"dem_{aoi.name}.tif"
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

    # ── terrain ────────────────────────────────────────────────────────────────
    if "terrain" in stages:
        log.info("--- Stage: terrain ---")
        from soilgeo.acquisition.sentinel_hub import fetch_dem
        from soilgeo.hydrology.whitebox import compute_twi_spi
        from soilgeo.terrain.derivatives import compute_curvature, compute_slope
        fetch_dem(bbox_wgs84=aoi.bbox, output_path=dem_path, resolution_m=30)
        compute_slope(dem_path, terrain_dir / "slope.tif")
        compute_curvature(dem_path, terrain_dir / "curvature.tif")
        compute_twi_spi(dem_path, hydro_dir / "twi.tif", hydro_dir / "spi.tif", work_dir=hydro_dir)

    # ── fetch_s2_seasonal ──────────────────────────────────────────────────────
    if "fetch_s2_seasonal" in stages:
        log.info("--- Stage: fetch_s2_seasonal ---")
        from soilgeo.acquisition.sentinel2 import fetch_s2_seasonal_indices
        s2_cfg = cfg["sentinel2_seasonal"]
        for season, win in s2_cfg["seasons"].items():
            fetch_s2_seasonal_indices(
                bbox_wgs84=aoi.bbox,
                time_interval=(win["start"], win["end"]),
                output_path=s2_dir / f"s2_{aoi.name}_{season}_ndvi_ndmi.tif",
                resolution_m=s2_cfg["resolution_m"],
            )

    # ── soilgrids (labels) ───────────────────────────────────────────────────
    if "soilgrids" in stages:
        log.info("--- Stage: soilgrids ---")
        from soilgeo.acquisition.soilgrids import download_soilgrids
        download_soilgrids(
            bbox=aoi.bbox, output_dir=sg_dir,
            properties=cfg["gbm"]["targets"], depth="0-5cm",
        )

    era5_nc = climate_dir / "era5_land_monthly.nc"

    # ── fetch_climate ───────────────────────────────────────────────────────────
    if "fetch_climate" in stages:
        log.info("--- Stage: fetch_climate ---")
        from soilgeo.acquisition.climate import download_era5_land
        cl = cfg["climate"]
        download_era5_land(
            bbox_wgs84=aoi.bbox, variables=cl["era5_land"]["variables"],
            year_start=int(cl["era5_land"]["start"][:4]),
            year_end=int(cl["era5_land"]["end"][:4]),
            output_path=era5_nc,
        )

    # ── climate_features: SPI-3 + temp/ET normals → rasters ──────────────────────
    if "climate_features" in stages:
        log.info("--- Stage: climate_features ---")
        import numpy as np
        import rasterio
        import xarray as xr
        from rasterio.transform import from_bounds

        from soilgeo.features.climate import spi_latest_raster
        cl = cfg["climate"]
        if not era5_nc.exists():
            log.warning("ERA5 NetCDF missing (%s) — skipping climate_features", era5_nc)
        else:
            ds = xr.open_dataset(era5_nc)
            # ERA5-Land monthly: dims (time/valid_time, lat, lon). Normalize names.
            latn = "latitude" if "latitude" in ds.dims else "lat"
            lonn = "longitude" if "longitude" in ds.dims else "lon"
            lats, lons = ds[latn].values, ds[lonn].values
            west, east = float(lons.min()), float(lons.max())
            south, north = float(lats.min()), float(lats.max())

            def _stack(var):
                a = np.asarray(ds[var].values, dtype=np.float64)  # (T, H, W)
                if lats[0] < lats[-1]:               # ensure north-up
                    a = a[:, ::-1, :]
                return a

            precip = _stack(cl["spi"]["precip_var"])
            spi3 = spi_latest_raster(precip, scale=cl["spi"]["scale_months"])
            temp_norm = np.nanmean(_stack("t2m"), axis=0)
            et_norm = np.nanmean(_stack("pev"), axis=0)
            precip_norm = np.nanmean(precip, axis=0)

            climate_dir.mkdir(parents=True, exist_ok=True)
            H, W = spi3.shape
            transform = from_bounds(west, south, east, north, W, H)
            for name, arr in [("spi3", spi3), ("temp_norm", temp_norm),
                              ("et_norm", et_norm), ("precip_norm", precip_norm)]:
                a = np.where(np.isfinite(arr), arr, NODATA).astype(np.float32)
                out = climate_dir / f"{name}.tif"
                with rasterio.open(
                    out, "w", driver="GTiff", height=H, width=W, count=1,
                    dtype="float32", crs="EPSG:4326", transform=transform,
                    nodata=NODATA, compress="deflate",
                ) as dst:
                    dst.write(a, 1)
                log.info("Climate feature written: %s (%dx%d)", out.name, W, H)
            ds.close()

    # Reference 10 m grid + feature/label source discovery (shared by build_cube/train).
    ref_grid = ts_dir / "vv_mean.tif"
    cube_work = interim / "_cube_work"
    matrix_dir = processed / "v3_features"

    def _split_band(src_path: Path, band: int, out_path: Path) -> Path:
        """Extract a single band to its own GeoTIFF (resample_to_match reads band 1)."""
        import rasterio
        if out_path.exists():
            return out_path
        with rasterio.open(src_path) as src:
            prof = src.profile.copy()
            prof.update(count=1)
            data = src.read(band)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(data, 1)
        return out_path

    def _discover_feature_paths() -> dict:
        feats = {}
        for p in sorted(ts_dir.glob("*.tif")):                  # SAR time-series stats
            feats[f"ts_{p.stem}"] = p
        for season in ("winter", "summer"):                     # S2 seasonal NDVI + NDMI
            s2p = s2_dir / f"s2_{aoi.name}_{season}_ndvi_ndmi.tif"
            if s2p.exists():
                feats[f"s2_{season}_ndvi"] = _split_band(
                    s2p, 1, cube_work / f"s2_{season}_ndvi.tif")
                feats[f"s2_{season}_ndmi"] = _split_band(
                    s2p, 2, cube_work / f"s2_{season}_ndmi.tif")
        for name, p in [("slope", terrain_dir / "slope.tif"),
                        ("curvature", terrain_dir / "curvature.tif"),
                        ("twi", hydro_dir / "twi.tif")]:
            if p.exists():
                feats[name] = p
        for cl_name in ("spi3", "temp_norm", "et_norm", "precip_norm"):  # optional climate
            p = climate_dir / f"{cl_name}.tif"
            if p.exists():
                feats[f"clim_{cl_name}"] = p
        return feats

    # ── build_cube ───────────────────────────────────────────────────────────
    if "build_cube" in stages:
        log.info("--- Stage: build_cube ---")
        import numpy as np

        from soilgeo.models.cube import assemble_cube, build_matrix, save_cube_zarr
        if not ref_grid.exists():
            log.warning("Reference grid missing (%s) — run ts_features first", ref_grid)
        else:
            feats = _discover_feature_paths()
            log.info("Feature sources (%d): %s", len(feats), sorted(feats))
            cube, names, _profile = assemble_cube(feats, ref_grid, cube_work)
            save_cube_zarr(cube, names, cube_path)

            sg = {t: sg_dir / f"sg_{t}_0_5cm_mean.tif" for t in cfg["gbm"]["targets"]}
            sg = {t: p for t, p in sg.items() if p.exists()}
            block_px = max(1, int(cfg["tiling"]["block_km"] * 1000 / aoi.resolution_m))
            X, y_dict, groups, valid = build_matrix(cube, sg, ref_grid, cube_work, block_px)

            matrix_dir.mkdir(parents=True, exist_ok=True)
            np.save(matrix_dir / "X.npy", X)
            np.save(matrix_dir / "groups.npy", groups)
            np.save(matrix_dir / "feat_names.npy", np.array(names))
            np.save(matrix_dir / "valid_mask.npy", valid)
            for t, y in y_dict.items():
                np.save(matrix_dir / f"y_{t}.npy", y)
            log.info("Saved training matrix: X=%s targets=%s", X.shape, list(y_dict))

    # ── train_gbm ────────────────────────────────────────────────────────────
    if "train_gbm" in stages:
        log.info("--- Stage: train_gbm ---")
        import numpy as np

        from soilgeo.models.gbm import predict_gbm, save_metrics, train_gbm_spatial_cv
        if not (matrix_dir / "X.npy").exists():
            log.warning("No training matrix — run build_cube first")
        else:
            import pickle
            X = np.load(matrix_dir / "X.npy")
            groups = np.load(matrix_dir / "groups.npy")
            names = list(np.load(matrix_dir / "feat_names.npy"))
            g = cfg["gbm"]
            models_dir = processed / "v3_gbm_models"
            models_dir.mkdir(parents=True, exist_ok=True)
            all_metrics = []
            for target in g["targets"]:
                yp = matrix_dir / f"y_{target}.npy"
                if not yp.exists():
                    continue
                y = np.load(yp)
                model, metrics = train_gbm_spatial_cv(
                    X, y, groups, names, target_name=target, backend=g["backend"],
                    n_estimators=g["n_estimators"], learning_rate=g["learning_rate"],
                    random_state=g["random_state"],
                )
                with open(models_dir / f"gbm_{target}.pkl", "wb") as f:
                    pickle.dump(model, f)
                all_metrics.append(metrics)
                _ = predict_gbm  # available for predict stage
            save_metrics(all_metrics, processed / "v3_gbm_metrics.json")

    # ── train_unet (regression head vs GBM baseline) ────────────────────────────
    if "train_unet" in stages:
        log.info("--- Stage: train_unet ---")
        import json

        import numpy as np
        import rasterio

        from soilgeo.models.cube import assemble_cube
        from soilgeo.models.dataset import SoilTileDataset, compute_norm_stats
        from soilgeo.models.tiling import assign_blocks, make_tile_index, split_blocks
        from soilgeo.models.train import evaluate_regression, train_unet
        from soilgeo.models.unet import UNet, resolve_device
        from soilgeo.utils.geo import resample_to_match

        u = cfg["unet"]
        target = u["regression_target"]
        if not ref_grid.exists():
            log.warning("No reference grid — run ts_features/build_cube first")
        else:
            import torch

            feats = _discover_feature_paths()
            cube, names, _p = assemble_cube(feats, ref_grid, cube_work)
            label_path = sg_dir / f"sg_{target}_0_5cm_mean.tif"
            aligned = cube_work / f"_label_{target}.tif"
            resample_to_match(label_path, ref_grid, aligned)
            label_raw = rasterio.open(aligned).read(1).astype(np.float32)
            # z-score the target for stable training; R² is scale-invariant so it
            # stays comparable to the GBM baseline. RMSE is converted back below.
            valid_lbl = label_raw != NODATA
            lbl_mean = float(label_raw[valid_lbl].mean())
            lbl_std = float(label_raw[valid_lbl].std())
            label = np.where(valid_lbl, (label_raw - lbl_mean) / lbl_std, NODATA).astype(np.float32)

            C, H, W = cube.shape
            t_cfg = cfg["tiling"]
            tiles = make_tile_index(H, W, tile=t_cfg["tile_px"], overlap=t_cfg["overlap_px"])
            blocks = assign_blocks(tiles, block_km=t_cfg["block_km"], pixel_m=aoi.resolution_m)
            split = split_blocks(blocks, tuple(t_cfg["split_ratios"]), seed=t_cfg["random_state"])

            def _tiles(idxs):
                return [tiles[i] for i in idxs]

            train_tiles = _tiles(split["train"])
            norm = compute_norm_stats([cube[:, r:r+h, c:c+w] for r, c, h, w in train_tiles])

            def _ds(idxs, aug):
                return SoilTileDataset(cube, label, _tiles(idxs), norm,
                                       tile_size=t_cfg["tile_px"], task="regression", augment=aug)

            from torch.utils.data import DataLoader
            bs = u["batch_size"]
            train_dl = DataLoader(_ds(split["train"], True), batch_size=bs, shuffle=True)
            val_dl = DataLoader(_ds(split["val"], False), batch_size=bs)
            test_dl = DataLoader(_ds(split["test"], False), batch_size=bs)

            model = UNet(in_channels=C, base_channels=u["base_channels"],
                         depth=u["depth"], n_classes=0, regression=True)
            train_unet(model, train_dl, val_dl, task="regression", epochs=u["epochs"],
                       lr=u["learning_rate"], weight_decay=u["weight_decay"],
                       device=u["device"], seed=u["random_state"])
            metrics = evaluate_regression(model, test_dl, resolve_device(u["device"]))
            metrics["target"] = target
            metrics["rmse"] = metrics["rmse"] * lbl_std        # back to g/kg
            metrics["mae"] = metrics["mae"] * lbl_std
            log.info("U-Net %s held-out R²=%.3f RMSE=%.2f g/kg (n=%d)",
                     target, metrics["r2"], metrics["rmse"], metrics["n"])

            models_dir = processed / "v3_unet_models"
            models_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), models_dir / f"unet_{target}.pt")
            norm.save(models_dir / f"unet_{target}_norm.json")
            (processed / "v3_unet_metrics.json").write_text(json.dumps([metrics], indent=2))

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
