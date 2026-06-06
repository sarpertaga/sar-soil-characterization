#!/usr/bin/env python
"""
V1 Exploratory Soil Characterization Pipeline.

Usage:
    conda activate soilgeo
    cp .env.example .env   # fill credentials
    python pipelines/run_v1.py
    python pipelines/run_v1.py --stages fetch_s1,terrain
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio
from dotenv import load_dotenv

from soilgeo.utils.config import load_aoi_config, load_pipeline_config
from soilgeo.utils.logging import get_logger

load_dotenv()

ALL_STAGES = [
    "download_dem",
    "fetch_s1",
    "terrain",
    "hydrology",
    "indices",
    "classify",
    "risk",
    "catalog",
]


def parse_args():
    p = argparse.ArgumentParser(description="V1 SAR soil characterization pipeline")
    p.add_argument("--config", default="config/pipelines/v1.yml")
    p.add_argument("--stages", default="all", help="Comma-separated stages or 'all'")
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    return p.parse_args()


def run(args):
    cfg = load_pipeline_config(Path(args.config))
    aoi = load_aoi_config(Path(cfg.aoi_config))
    log = get_logger("pipeline.v1", log_dir=Path(cfg.paths["log_dir"]))

    raw = Path(cfg.paths["raw_dir"])
    interim = Path(cfg.paths["interim_dir"])
    processed = Path(cfg.paths["processed_dir"])
    stages = ALL_STAGES if args.stages == "all" else args.stages.split(",")

    log.info("=== V1 Pipeline | AOI: %s | stages: %s ===", aoi.name, stages)

    # ── download_dem ────────────────────────────────────────────────────────
    if "download_dem" in stages:
        log.info("--- Stage: download_dem ---")
        from soilgeo.acquisition.dem import download_dem_tiles, mosaic_and_reproject
        tiles = download_dem_tiles(aoi.bbox, raw / "dem", buffer_deg=aoi.dem["buffer_deg"])
        mosaic_and_reproject(
            tiles,
            interim / "dem" / f"dem_{aoi.name}_30m.tif",
            target_crs=aoi.crs,
            resolution_m=30,
        )

    # ── fetch_s1 ────────────────────────────────────────────────────────────
    if "fetch_s1" in stages:
        log.info("--- Stage: fetch_s1 ---")
        from soilgeo.acquisition.sentinel_hub import fetch_backscatter
        s1_cfg = aoi.sentinel1
        for season in ("wet", "dry"):
            season_cfg = s1_cfg[f"{season}_season"]
            out = interim / "s1" / f"s1_vvvh_{aoi.name}_{season}_median.tif"
            fetch_backscatter(
                bbox_wgs84=aoi.bbox,
                time_interval=(season_cfg["start"], season_cfg["end"]),
                output_path=out,
                resolution_m=aoi.resolution_m,
                median_composite=True,
            )

    # ── terrain ─────────────────────────────────────────────────────────────
    if "terrain" in stages:
        log.info("--- Stage: terrain ---")
        from soilgeo.terrain.derivatives import (
            compute_slope, compute_aspect, compute_roughness,
            compute_hillshade, compute_curvature,
        )
        dem = interim / "dem" / f"dem_{aoi.name}_30m.tif"
        t = interim / "terrain"
        compute_slope(dem, t / "slope.tif")
        compute_aspect(dem, t / "aspect.tif")
        compute_roughness(dem, t / "roughness.tif")
        compute_hillshade(dem, t / "hillshade.tif")
        compute_curvature(dem, t / "curvature.tif")

    # ── hydrology ───────────────────────────────────────────────────────────
    if "hydrology" in stages:
        log.info("--- Stage: hydrology ---")
        from soilgeo.hydrology.whitebox import compute_flow_accumulation, compute_twi_spi
        dem = interim / "dem" / f"dem_{aoi.name}_30m.tif"
        h = interim / "hydrology"
        compute_flow_accumulation(dem, h / "flow_acc.tif", work_dir=h)
        compute_twi_spi(dem, h / "twi.tif", h / "spi.tif", work_dir=h)

    # ── indices ─────────────────────────────────────────────────────────────
    if "indices" in stages:
        log.info("--- Stage: indices ---")
        from soilgeo.indices.sar import compute_vv_vh_ratio, compute_moisture_index
        s1 = interim / "s1"
        idx = interim / "indices"

        def _split_bands(composite_path: Path, season: str):
            vv_out = s1 / f"s1_vv_{aoi.name}_{season}.tif"
            vh_out = s1 / f"s1_vh_{aoi.name}_{season}.tif"
            if not vv_out.exists():
                with rasterio.open(composite_path) as src:
                    profile = src.profile.copy()
                    profile.update(count=1)
                    for band, path in [(1, vv_out), (2, vh_out)]:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with rasterio.open(path, "w", **profile) as dst:
                            dst.write(src.read(band), 1)
            return vv_out, vh_out

        wet_composite = s1 / f"s1_vvvh_{aoi.name}_wet_median.tif"
        dry_composite = s1 / f"s1_vvvh_{aoi.name}_dry_median.tif"
        vv_wet, vh_wet = _split_bands(wet_composite, "wet")
        vv_dry, vh_dry = _split_bands(dry_composite, "dry")

        compute_vv_vh_ratio(vv_wet, vh_wet, idx / "vv_vh_ratio_wet.tif")
        compute_vv_vh_ratio(vv_dry, vh_dry, idx / "vv_vh_ratio_dry.tif")
        compute_moisture_index(vv_wet, vv_dry, vv_wet, idx / "moisture_index.tif")

    # ── classify ────────────────────────────────────────────────────────────
    if "classify" in stages:
        log.info("--- Stage: classify ---")
        from soilgeo.analysis.classification import classify_surface_response
        mi_path = interim / "indices" / "moisture_index.tif"
        twi_path = interim / "hydrology" / "twi.tif"
        slope_path = interim / "terrain" / "slope.tif"

        if not all(p.exists() for p in [mi_path, twi_path, slope_path]):
            log.warning("Missing inputs for classify — skipping")
        else:
            with rasterio.open(mi_path) as src:
                mi = src.read(1).flatten().astype(np.float32)
                profile = src.profile
                h, w = src.height, src.width
            with rasterio.open(twi_path) as src:
                twi_r = src.read(1).flatten().astype(np.float32)
            with rasterio.open(slope_path) as src:
                slope_r = src.read(1).flatten().astype(np.float32)

            src_cfg = cfg.surface_response_classes
            labels = classify_surface_response(
                {"moisture_index": mi, "twi": twi_r, "slope": slope_r},
                n_clusters=src_cfg["n_clusters"],
                random_state=src_cfg["random_state"],
            ).reshape(h, w)

            out = processed / f"surface_response_classes_{aoi.name}.tif"
            out.parent.mkdir(parents=True, exist_ok=True)
            profile.update(dtype="uint8", nodata=255, count=1, compress="deflate")
            with rasterio.open(out, "w", **profile) as dst:
                dst.write(labels, 1)
            log.info("Surface Response Classes written: %s", out.name)

    # ── risk ────────────────────────────────────────────────────────────────
    if "risk" in stages and cfg.construction_risk["enabled"]:
        log.info("--- Stage: risk ---")
        from soilgeo.products.cog import compute_construction_risk
        mi_path = interim / "indices" / "moisture_index.tif"
        fa_path = interim / "hydrology" / "flow_acc.tif"
        slope_path = interim / "terrain" / "slope.tif"
        if all(p.exists() for p in [mi_path, fa_path, slope_path]):
            compute_construction_risk(
                mi_path=mi_path,
                flow_acc_path=fa_path,
                slope_path=slope_path,
                output_path=processed / f"construction_risk_{aoi.name}.tif",
                weights=cfg.construction_risk["weights"],
            )

    # ── catalog ─────────────────────────────────────────────────────────────
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
