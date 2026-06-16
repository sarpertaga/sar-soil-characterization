#!/usr/bin/env python
"""
V1 Exploratory Soil Characterization Pipeline.

All raster data fetched via Sentinel Hub API — no raw downloads.
Only processed GeoTIFFs are saved locally.

Usage:
    conda activate soilgeo
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
    "fetch_dem",    # Copernicus DEM elevation via Sentinel Hub
    "fetch_s1",     # Sentinel-1 wet + dry composites via Sentinel Hub
    "terrain",      # slope, aspect, curvature, hillshade from DEM TIF
    "hydrology",    # TWI, SPI, flow accumulation from DEM TIF
    "indices",      # VV/VH ratio, NDDI
    "classify",     # Surface Response Classes (k-means)
    "stats",        # Kruskal-Wallis + Spearman analysis
    "risk",         # Construction Moisture Risk Index
    "catalog",      # Product catalog JSON
]


def parse_args():
    p = argparse.ArgumentParser(description="V1 SAR soil characterization pipeline")
    p.add_argument("--config", default="config/pipelines/v1.yml")
    p.add_argument("--stages", default="all", help="Comma-separated stages or 'all'")
    return p.parse_args()


def run(args):
    cfg = load_pipeline_config(Path(args.config))
    aoi = load_aoi_config(Path(cfg.aoi_config))
    log = get_logger("pipeline.v1", log_dir=Path(cfg.paths["log_dir"]))

    interim = Path(cfg.paths["interim_dir"]).resolve()
    processed = Path(cfg.paths["processed_dir"]).resolve()
    stages = ALL_STAGES if args.stages == "all" else args.stages.split(",")

    log.info("=== V1 Pipeline | AOI: %s | stages: %s ===", aoi.name, stages)

    dem_path = interim / "dem" / f"dem_{aoi.name}_30m.tif"

    # ── fetch_dem ────────────────────────────────────────────────────────────
    if "fetch_dem" in stages:
        log.info("--- Stage: fetch_dem ---")
        from soilgeo.acquisition.sentinel_hub import fetch_dem
        fetch_dem(
            bbox_wgs84=aoi.bbox,
            output_path=dem_path,
            resolution_m=30,
        )

    # ── fetch_s1 ────────────────────────────────────────────────────────────
    if "fetch_s1" in stages:
        log.info("--- Stage: fetch_s1 ---")
        from soilgeo.acquisition.sentinel_hub import fetch_backscatter
        s1_cfg = aoi.sentinel1
        for season in ("wet", "dry"):
            season_cfg = s1_cfg[f"{season}_season"]
            fetch_backscatter(
                bbox_wgs84=aoi.bbox,
                time_interval=(season_cfg["start"], season_cfg["end"]),
                output_path=interim / "s1" / f"s1_vvvh_{aoi.name}_{season}_median.tif",
                resolution_m=aoi.resolution_m,
                median_composite=True,
            )

    # ── terrain ─────────────────────────────────────────────────────────────
    if "terrain" in stages:
        log.info("--- Stage: terrain ---")
        from soilgeo.terrain.derivatives import (
            compute_aspect,
            compute_curvature,
            compute_hillshade,
            compute_roughness,
            compute_slope,
        )
        t = interim / "terrain"
        compute_slope(dem_path, t / "slope.tif")
        compute_aspect(dem_path, t / "aspect.tif")
        compute_roughness(dem_path, t / "roughness.tif")
        compute_hillshade(dem_path, t / "hillshade.tif")
        compute_curvature(dem_path, t / "curvature.tif")

    # ── hydrology ───────────────────────────────────────────────────────────
    if "hydrology" in stages:
        log.info("--- Stage: hydrology ---")
        from soilgeo.hydrology.whitebox import compute_flow_accumulation, compute_twi_spi
        h = interim / "hydrology"
        compute_flow_accumulation(dem_path, h / "flow_acc.tif", work_dir=h)
        compute_twi_spi(dem_path, h / "twi.tif", h / "spi.tif", work_dir=h)

    # ── indices ─────────────────────────────────────────────────────────────
    if "indices" in stages:
        log.info("--- Stage: indices ---")
        from soilgeo.indices.sar import compute_nddi, compute_vv_vh_ratio
        s1 = interim / "s1"
        idx = interim / "indices"

        def _split(composite: Path, season: str):
            vv_out = s1 / f"s1_vv_{aoi.name}_{season}.tif"
            vh_out = s1 / f"s1_vh_{aoi.name}_{season}.tif"
            if not vv_out.exists():
                with rasterio.open(composite) as src:
                    profile = src.profile.copy()
                    profile.update(count=1)
                    for band, path in [(1, vv_out), (2, vh_out)]:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with rasterio.open(path, "w", **profile) as dst:
                            dst.write(src.read(band), 1)
            return vv_out, vh_out

        vv_wet, vh_wet = _split(s1 / f"s1_vvvh_{aoi.name}_wet_median.tif", "wet")
        vv_dry, vh_dry = _split(s1 / f"s1_vvvh_{aoi.name}_dry_median.tif", "dry")

        compute_vv_vh_ratio(vv_wet, vh_wet, idx / "vv_vh_ratio_wet.tif")
        compute_vv_vh_ratio(vv_dry, vh_dry, idx / "vv_vh_ratio_dry.tif")
        compute_nddi(vv_wet, vv_dry, idx / "nddi.tif")

    # ── classify ────────────────────────────────────────────────────────────
    if "classify" in stages:
        log.info("--- Stage: classify ---")
        from soilgeo.analysis.classification import classify_surface_response
        from soilgeo.utils.geo import resample_to_match
        nddi_path = interim / "indices" / "nddi.tif"
        twi_path = interim / "hydrology" / "twi.tif"
        slope_path = interim / "terrain" / "slope.tif"

        # Resample terrain layers to match S1 grid (NDDI is the reference)
        twi_r_path = interim / "hydrology" / "twi_resampled.tif"
        slope_r_path = interim / "terrain" / "slope_resampled.tif"
        resample_to_match(twi_path, nddi_path, twi_r_path)
        resample_to_match(slope_path, nddi_path, slope_r_path)

        if not all(p.exists() for p in [nddi_path, twi_r_path, slope_r_path]):
            log.warning("Missing inputs for classify — skipping")
        else:
            with rasterio.open(nddi_path) as src:
                mi = src.read(1).flatten().astype(np.float32)
                profile = src.profile
                h, w = src.height, src.width
            with rasterio.open(twi_r_path) as src:
                twi_r = src.read(1).flatten().astype(np.float32)
            with rasterio.open(slope_r_path) as src:
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
            log.info("Surface Response Classes: %s", out.name)

    # ── stats ────────────────────────────────────────────────────────────────
    if "stats" in stages:
        log.info("--- Stage: stats ---")
        from soilgeo.analysis.report import run_stats_analysis
        from soilgeo.utils.geo import resample_to_match
        vv_wet_path   = interim / "s1"      / f"s1_vv_{aoi.name}_wet.tif"
        vv_dry_path   = interim / "s1"      / f"s1_vv_{aoi.name}_dry.tif"
        nddi_path     = interim / "indices" / "nddi.tif"
        twi_path      = interim / "hydrology" / "twi.tif"
        slope_path    = interim / "terrain"   / "slope.tif"
        twi_r_path    = interim / "hydrology" / "twi_resampled.tif"
        slope_r_path  = interim / "terrain"   / "slope_resampled.tif"
        # Ensure resampled versions exist
        resample_to_match(twi_path, nddi_path, twi_r_path)
        resample_to_match(slope_path, nddi_path, slope_r_path)
        run_stats_analysis(
            vv_wet_path=vv_wet_path,
            vv_dry_path=vv_dry_path,
            nddi_path=nddi_path,
            twi_path=twi_r_path,
            slope_path=slope_r_path,
            output_path=processed / "stats_v1.json",
        )

    # ── risk ────────────────────────────────────────────────────────────────
    if "risk" in stages and cfg.construction_risk["enabled"]:
        log.info("--- Stage: risk ---")
        from soilgeo.products.cog import compute_construction_risk
        from soilgeo.utils.geo import resample_to_match
        fa_path = interim / "hydrology" / "flow_acc.tif"
        slope_path = interim / "terrain" / "slope.tif"
        nddi_path = interim / "indices" / "nddi.tif"
        fa_r_path = interim / "hydrology" / "flow_acc_resampled.tif"
        slope_risk_path = interim / "terrain" / "slope_resampled.tif"
        resample_to_match(fa_path, nddi_path, fa_r_path)
        if all(p.exists() for p in [nddi_path, fa_r_path, slope_risk_path]):
            compute_construction_risk(
                mi_path=nddi_path,
                flow_acc_path=fa_r_path,
                slope_path=slope_risk_path,
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
