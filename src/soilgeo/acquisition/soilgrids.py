"""Download SoilGrids v2.0 rasters via ISRIC WCS for a given bbox."""
from pathlib import Path

import requests

from soilgeo.utils.logging import get_logger

log = get_logger(__name__)

_WCS_BASE = "https://maps.isric.org/mapserv"

# SoilGrids properties available via WCS, with native units.
SOILGRIDS_PROPERTIES = {
    "clay":  {"map": "clay.map",  "unit": "g/kg",  "scale_to_pct": 0.1},
    "sand":  {"map": "sand.map",  "unit": "g/kg",  "scale_to_pct": 0.1},
    "silt":  {"map": "silt.map",  "unit": "g/kg",  "scale_to_pct": 0.1},
    "soc":   {"map": "soc.map",   "unit": "dg/kg", "scale_to_pct": None},
    "bdod":  {"map": "bdod.map",  "unit": "cg/cm³","scale_to_pct": None},
}

VALID_DEPTHS = ["0-5cm", "5-15cm", "15-30cm"]


def build_wcs_url(
    prop: str,
    depth: str,
    west: float,
    south: float,
    east: float,
    north: float,
) -> str:
    """Build a SoilGrids WCS GetCoverage URL for a property / depth / bbox."""
    if prop not in SOILGRIDS_PROPERTIES:
        raise ValueError(f"Unknown property '{prop}'. Choose from: {list(SOILGRIDS_PROPERTIES)}")
    if depth not in VALID_DEPTHS:
        raise ValueError(f"Unknown depth '{depth}'. Choose from: {VALID_DEPTHS}")

    map_file = SOILGRIDS_PROPERTIES[prop]["map"]
    coverage_id = f"{prop}_{depth}_mean"

    # urllib.parse.urlencode doesn't handle repeated keys well; build manually
    return (
        f"{_WCS_BASE}?map=/map/{map_file}"
        f"&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
        f"&COVERAGEID={coverage_id}"
        f"&FORMAT=image/tiff"
        f"&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
        f"&SUBSET=X({west},{east})"
        f"&SUBSET=Y({south},{north})"
    )


def download_soilgrids(
    bbox: dict,
    output_dir: Path,
    properties: list[str] | None = None,
    depth: str = "0-5cm",
) -> dict[str, Path]:
    """
    Download SoilGrids mean rasters for the given bbox.

    Args:
        bbox:       dict with keys west/south/east/north (EPSG:4326)
        output_dir: directory to save downloaded TIFFs
        properties: list of property names; defaults to ["clay","sand","soc"]
        depth:      soil depth horizon (default "0-5cm")

    Returns:
        dict mapping property name → local Path
    """
    if properties is None:
        properties = ["clay", "sand", "soc"]

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for prop in properties:
        out = output_dir / f"sg_{prop}_{depth.replace('-','_')}_mean.tif"
        if out.exists():
            log.info("SoilGrids already downloaded: %s", out.name)
            paths[prop] = out
            continue

        url = build_wcs_url(
            prop=prop, depth=depth,
            west=bbox["west"], south=bbox["south"],
            east=bbox["east"], north=bbox["north"],
        )
        log.info("Downloading SoilGrids %s %s …", prop, depth)
        try:
            resp = requests.get(url, timeout=60, headers={"User-Agent": "soilgeo/2.0"})
            resp.raise_for_status()
            out.write_bytes(resp.content)
            log.info("  → saved: %s (%.0f KB)", out.name, len(resp.content) / 1024)
            paths[prop] = out
        except Exception as exc:
            log.error("Failed to download SoilGrids %s: %s", prop, exc)

    return paths
