#!/usr/bin/env python
"""
V2 Soil Property Modelling Pipeline.

Extends V1 with Sentinel-2 optical features and SoilGrids ground truth
to produce 10 m clay, sand, and SOC prediction maps.

Usage:
    conda activate soilgeo
    python pipelines/run_v2.py
    python pipelines/run_v2.py --stages fetch_s2,s2_indices
    python pipelines/run_v2.py --stages soilgrids,features,train,predict
"""
import argparse
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from soilgeo.utils.config import load_aoi_config, load_config_dict
from soilgeo.utils.logging import get_logger

load_dotenv()

ALL_STAGES = [
    "fetch_s2",    # Sentinel-2 bare-soil composite (Sentinel Hub)
    "s2_indices",  # BSI, clay_index, NDVI, NDWI, iron_oxide
    "soilgrids",   # Download SoilGrids clay/sand/SOC targets
    "features",    # Build training matrix at SoilGrids resolution
    "train",       # Random Forest CV training for each target
    "predict",     # Predict at 10 m, write GeoTIFFs
    "catalog",     # Update product catalog
]


def parse_args():
    p = argparse.ArgumentParser(description="V2 soil property modelling pipeline")
    p.add_argument("--config", default="config/pipelines/v2.yml")
    p.add_argument("--stages", default="all", help="Comma-separated stages or 'all'")
    return p.parse_args()


def run(args):  # noqa: C901
    cfg = load_config_dict(Path(args.config))
    aoi = load_aoi_config(Path(cfg["aoi_config"]))
    log = get_logger("pipeline.v2", log_dir=Path(cfg["paths"]["log_dir"]))

    interim   = Path(cfg["paths"]["interim_dir"]).resolve()
    processed = Path(cfg["paths"]["processed_dir"]).resolve()
    stages    = ALL_STAGES if args.stages == "all" else args.stages.split(",")

    log.info("=== V2 Pipeline | AOI: %s | stages: %s ===", aoi.name, stages)

    s2_cfg   = cfg["sentinel2"]
    sg_cfg   = cfg["soilgrids"]
    ml_cfg   = cfg["modelling"]
    rf_cfg   = ml_cfg["random_forest"]

    # Paths — V1 outputs (required by features stage)
    v1 = {
        "vv_wet":    interim / "s1"       / f"s1_vv_{aoi.name}_wet.tif",
        "vh_wet":    interim / "s1"       / f"s1_vh_{aoi.name}_wet.tif",
        "vv_dry":    interim / "s1"       / f"s1_vv_{aoi.name}_dry.tif",
        "vh_dry":    interim / "s1"       / f"s1_vh_{aoi.name}_dry.tif",
        "nddi":      interim / "indices"  / "nddi.tif",
        "slope":     interim / "terrain"  / "slope_resampled.tif",
        "twi":       interim / "hydrology"/ "twi_resampled.tif",
        "curvature": interim / "terrain"  / "curvature.tif",
    }

    # Paths — V2 outputs
    s2_dir  = interim / "s2"
    idx_dir = interim / "indices_s2"
    sg_dir  = interim / "soilgrids"
    s2_composite = s2_dir / f"s2_bare_soil_{aoi.name}.tif"
    s2_indices = {
        "bsi":        idx_dir / f"s2_bare_soil_{aoi.name}_bsi.tif",
        "clay_index": idx_dir / f"s2_bare_soil_{aoi.name}_clay_index.tif",
        "ndvi":       idx_dir / f"s2_bare_soil_{aoi.name}_ndvi.tif",
        "ndwi":       idx_dir / f"s2_bare_soil_{aoi.name}_ndwi.tif",
        "iron_oxide": idx_dir / f"s2_bare_soil_{aoi.name}_iron_oxide.tif",
    }

    # ── fetch_s2 ─────────────────────────────────────────────────────────────
    if "fetch_s2" in stages:
        log.info("--- Stage: fetch_s2 ---")
        from soilgeo.acquisition.sentinel2 import fetch_s2_bare_soil
        s2_season = s2_cfg["bare_soil_season"]
        fetch_s2_bare_soil(
            bbox_wgs84=aoi.bbox,
            time_interval=(s2_season["start"], s2_season["end"]),
            output_path=s2_composite,
            resolution_m=s2_cfg["resolution_m"],
            ndvi_threshold=s2_cfg["ndvi_threshold"],
        )

    # ── s2_indices ───────────────────────────────────────────────────────────
    if "s2_indices" in stages:
        log.info("--- Stage: s2_indices ---")
        from soilgeo.indices.optical import compute_all_indices
        if not s2_composite.exists():
            log.warning("S2 composite missing — skipping s2_indices")
        else:
            idx_dir.mkdir(parents=True, exist_ok=True)
            computed = compute_all_indices(s2_composite, idx_dir)
            log.info("S2 indices written: %s", list(computed.keys()))

    # ── soilgrids ────────────────────────────────────────────────────────────
    if "soilgrids" in stages:
        log.info("--- Stage: soilgrids ---")
        from soilgeo.acquisition.soilgrids import download_soilgrids
        download_soilgrids(
            bbox=aoi.bbox,
            output_dir=sg_dir,
            properties=sg_cfg["properties"],
            depth=sg_cfg["depth"],
        )

    # ── features ─────────────────────────────────────────────────────────────
    if "features" in stages:
        log.info("--- Stage: features ---")
        from soilgeo.modelling.features import build_prediction_stack, build_training_matrix
        from soilgeo.utils.geo import resample_to_match

        # Resample curvature to match NDDI grid if needed
        curv_src = interim / "terrain" / "curvature.tif"
        resample_to_match(curv_src, v1["nddi"], v1["curvature"])

        feature_paths = {**v1, **s2_indices}
        missing = [k for k, p in feature_paths.items() if not p.exists()]
        if missing:
            log.warning("Missing feature rasters: %s — skipping features", missing)
        else:
            sg_paths = {
                prop: sg_dir / f"sg_{prop}_{sg_cfg['depth'].replace('-','_')}_mean.tif"
                for prop in sg_cfg["properties"]
            }
            missing_sg = [k for k, p in sg_paths.items() if not p.exists()]
            if missing_sg:
                log.warning("Missing SoilGrids rasters: %s — skipping features", missing_sg)
            else:
                X, y_dict, feat_names = build_training_matrix(feature_paths, sg_paths)
                # Save feature matrix for reproducibility
                matrix_dir = processed / "v2_features"
                matrix_dir.mkdir(parents=True, exist_ok=True)
                np.save(matrix_dir / "X_train.npy", X)
                np.save(matrix_dir / "feat_names.npy", np.array(feat_names))
                for name, y in y_dict.items():
                    np.save(matrix_dir / f"y_{name}.npy", y)
                log.info("Feature matrix saved: X=%s", X.shape)

    # ── train ────────────────────────────────────────────────────────────────
    if "train" in stages:
        log.info("--- Stage: train ---")
        import pickle

        from soilgeo.modelling.regression import save_metrics, train_soil_model

        matrix_dir = processed / "v2_features"
        X = np.load(matrix_dir / "X_train.npy")
        feat_names = list(np.load(matrix_dir / "feat_names.npy"))
        models_dir = processed / "v2_models"
        models_dir.mkdir(parents=True, exist_ok=True)
        all_metrics = []

        for prop in sg_cfg["properties"]:
            y_path = matrix_dir / f"y_{prop}.npy"
            if not y_path.exists():
                log.warning("No training labels for %s — skipping", prop)
                continue
            y = np.load(y_path)
            model, metrics = train_soil_model(
                X, y, feat_names, target_name=prop,
                n_estimators=rf_cfg["n_estimators"],
                cv_folds=rf_cfg["cv_folds"],
                random_state=rf_cfg["random_state"],
            )
            model_path = models_dir / f"rf_{prop}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            log.info("Model saved: %s", model_path.name)
            all_metrics.append(metrics)

        save_metrics(all_metrics, processed / "v2_metrics.json")

    # ── predict ──────────────────────────────────────────────────────────────
    if "predict" in stages:
        log.info("--- Stage: predict ---")
        import pickle

        from soilgeo.modelling.features import build_prediction_stack
        from soilgeo.modelling.regression import predict_raster

        feature_paths = {**v1, **s2_indices}
        missing = [k for k, p in feature_paths.items() if not p.exists()]
        if missing:
            log.warning("Missing feature rasters for prediction: %s", missing)
        else:
            X_full, profile, shape = build_prediction_stack(feature_paths)
            models_dir = processed / "v2_models"

            for prop in sg_cfg["properties"]:
                model_path = models_dir / f"rf_{prop}.pkl"
                if not model_path.exists():
                    log.warning("No model for %s — skipping", prop)
                    continue
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                predict_raster(
                    model, X_full, shape, profile,
                    output_path=processed / f"{prop}_10m_{aoi.name}.tif",
                )

    # ── catalog ──────────────────────────────────────────────────────────────
    if "catalog" in stages:
        log.info("--- Stage: catalog ---")
        from soilgeo.products.cog import write_product_catalog
        entries = [
            {"name": p.stem, "path": str(p.relative_to(processed)), "type": "COG"}
            for p in sorted(processed.glob("*.tif"))
        ]
        write_product_catalog(processed, entries)

    log.info("=== V2 Pipeline complete ===")


if __name__ == "__main__":
    run(parse_args())
