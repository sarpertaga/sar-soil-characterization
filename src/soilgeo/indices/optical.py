"""Soil-relevant optical indices derived from Sentinel-2 L2A bare-soil composite."""
from pathlib import Path

import numpy as np
import rasterio

from soilgeo.utils.logging import get_logger

log = get_logger(__name__)

NODATA = -9999.0

# Band positions in the 7-band S2 composite GeoTIFF (1-based rasterio convention)
B02, B03, B04, B08, B8A, B11, B12 = 1, 2, 3, 4, 5, 6, 7


def _read_s2_bands(s2_path: Path) -> tuple[dict[str, np.ndarray], dict, np.ndarray]:
    """Read all 7 S2 bands and the valid mask. Returns (bands_dict, profile, nodata_mask)."""
    with rasterio.open(s2_path) as src:
        profile = src.profile.copy()
        raw = {
            "B02": src.read(B02).astype(np.float32),
            "B03": src.read(B03).astype(np.float32),
            "B04": src.read(B04).astype(np.float32),
            "B08": src.read(B08).astype(np.float32),
            "B8A": src.read(B8A).astype(np.float32),
            "B11": src.read(B11).astype(np.float32),
            "B12": src.read(B12).astype(np.float32),
        }
        nd = src.nodata if src.nodata is not None else NODATA
    nodata_mask = np.any(np.stack(list(raw.values()), axis=0) == nd, axis=0)
    return raw, profile, nodata_mask


def _write_index(
    arr: np.ndarray,
    nodata_mask: np.ndarray,
    profile: dict,
    output_path: Path,
    name: str,
) -> Path:
    arr = np.where(nodata_mask, NODATA, arr).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    p = profile.copy()
    p.update(count=1, dtype="float32", nodata=NODATA, compress="deflate")
    with rasterio.open(output_path, "w", **p) as dst:
        dst.write(arr, 1)
    valid = arr[arr != NODATA]
    log.info("%s written: range [%.4f, %.4f] mean=%.4f", name,
             float(valid.min()), float(valid.max()), float(valid.mean()))
    return output_path


def _normalized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(a + b) < 1e-9, 0.0, (a - b) / (a + b))


def compute_bsi(s2_path: Path, output_path: Path) -> Path:
    """
    Bare Soil Index: BSI = ((B11+B04) - (B08+B02)) / ((B11+B04) + (B08+B02))
    Range ~[-1, 1]. Higher = more bare soil.
    """
    if output_path.exists():
        log.info("Skipping BSI (exists)")
        return output_path
    b, profile, nd_mask = _read_s2_bands(s2_path)
    num = (b["B11"] + b["B04"]) - (b["B08"] + b["B02"])
    den = (b["B11"] + b["B04"]) + (b["B08"] + b["B02"])
    with np.errstate(divide="ignore", invalid="ignore"):
        bsi = np.where(np.abs(den) < 1e-9, 0.0, num / den)
    return _write_index(bsi, nd_mask, profile, output_path, "BSI")


def compute_clay_index(s2_path: Path, output_path: Path) -> Path:
    """
    Clay Mineral Index: CI = B11 / B12 (SWIR1 / SWIR2).
    Sensitive to clay, illite, kaolinite absorption features.
    """
    if output_path.exists():
        log.info("Skipping Clay Index (exists)")
        return output_path
    b, profile, nd_mask = _read_s2_bands(s2_path)
    with np.errstate(divide="ignore", invalid="ignore"):
        ci = np.where(np.abs(b["B12"]) < 1e-9, NODATA, b["B11"] / b["B12"])
    return _write_index(ci, nd_mask, profile, output_path, "ClayIndex")


def compute_ndvi(s2_path: Path, output_path: Path) -> Path:
    """NDVI = (B08 - B04) / (B08 + B04). Used for vegetation masking and SOC proxy."""
    if output_path.exists():
        log.info("Skipping NDVI (exists)")
        return output_path
    b, profile, nd_mask = _read_s2_bands(s2_path)
    ndvi = _normalized(b["B08"], b["B04"])
    return _write_index(ndvi, nd_mask, profile, output_path, "NDVI")


def compute_ndwi(s2_path: Path, output_path: Path) -> Path:
    """NDWI = (B03 - B08) / (B03 + B08). Sensitive to soil moisture."""
    if output_path.exists():
        log.info("Skipping NDWI (exists)")
        return output_path
    b, profile, nd_mask = _read_s2_bands(s2_path)
    ndwi = _normalized(b["B03"], b["B08"])
    return _write_index(ndwi, nd_mask, profile, output_path, "NDWI")


def compute_iron_oxide(s2_path: Path, output_path: Path) -> Path:
    """
    Iron Oxide Ratio: IOR = B04 / B02.
    Higher = iron-rich / reddish soils.
    """
    if output_path.exists():
        log.info("Skipping Iron Oxide (exists)")
        return output_path
    b, profile, nd_mask = _read_s2_bands(s2_path)
    with np.errstate(divide="ignore", invalid="ignore"):
        ior = np.where(np.abs(b["B02"]) < 1e-9, NODATA, b["B04"] / b["B02"])
    return _write_index(ior, nd_mask, profile, output_path, "IronOxide")


def compute_all_indices(s2_path: Path, output_dir: Path) -> dict[str, Path]:
    """Compute all soil optical indices from a single S2 composite GeoTIFF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = s2_path.stem
    return {
        "bsi":         compute_bsi(s2_path, output_dir / f"{stem}_bsi.tif"),
        "clay_index":  compute_clay_index(s2_path, output_dir / f"{stem}_clay_index.tif"),
        "ndvi":        compute_ndvi(s2_path, output_dir / f"{stem}_ndvi.tif"),
        "ndwi":        compute_ndwi(s2_path, output_dir / f"{stem}_ndwi.tif"),
        "iron_oxide":  compute_iron_oxide(s2_path, output_dir / f"{stem}_iron_oxide.tif"),
    }
