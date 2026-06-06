import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer


NODATA = -9999.0


def write_cog(
    path: Path,
    data: np.ndarray,
    profile: dict,
    nodata: float = NODATA,
    provenance: dict | None = None,
) -> Path:
    profile = profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=nodata,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        interleave="band",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)
        dst.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")

    if provenance is not None:
        provenance["_written_at"] = datetime.now(timezone.utc).isoformat()
        path.with_suffix(".json").write_text(json.dumps(provenance, indent=2))

    return path


def bbox_to_utm(west: float, south: float, east: float, north: float, crs: str) -> dict:
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    minx, miny = transformer.transform(west, south)
    maxx, maxy = transformer.transform(east, north)
    return {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy}


def read_band(path: Path, band: int = 1) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(band).astype(np.float32)
