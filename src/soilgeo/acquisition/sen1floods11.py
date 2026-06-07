"""
Sen1Floods11 hand-labeled subset downloader (Phase 2 U-Net showcase).

Sen1Floods11 (Bonafilia et al., 2020, CVPRW) is a public flood-mapping benchmark
hosted on the GCS bucket ``gs://sen1floods11``. We use the **hand-labeled**
subset: 446 chips of 512×512 px, each a Sentinel-1 VV+VH GeoTIFF (``S1Hand``)
paired with a hand-drawn flood mask (``LabelHand``: -1 nodata, 0 not-water,
1 water). This is the dense-pixel-mask, texture-driven SAR task where a U-Net is
expected to beat a per-pixel baseline — the complement to the soil task where it
did not.

The bucket is public, so objects are listed via the JSON API and fetched over
plain HTTPS (no credentials).
"""
import urllib.parse
import urllib.request
from pathlib import Path

from soilgeo.utils.logging import get_logger

log = get_logger(__name__)

_BUCKET = "sen1floods11"
_LIST = "https://storage.googleapis.com/storage/v1/b/{b}/o"
_DL = "https://storage.googleapis.com/{b}/{name}"
_PREFIX = "v1.1/data/flood_events/HandLabeled"


def _list_objects(prefix: str) -> list[str]:
    """Return all object names under a bucket prefix (paginated JSON API)."""
    import json

    names, token = [], None
    while True:
        url = f"{_LIST.format(b=_BUCKET)}?prefix={urllib.parse.quote(prefix)}&maxResults=1000"
        if token:
            url += f"&pageToken={token}"
        with urllib.request.urlopen(url) as r:
            data = json.load(r)
        names += [it["name"] for it in data.get("items", [])]
        token = data.get("nextPageToken")
        if not token:
            break
    return [n for n in names if n.endswith(".tif")]


def download_handlabeled(out_dir: Path, limit: int | None = None) -> dict[str, list[Path]]:
    """
    Download the hand-labeled S1 chips + flood masks. Resumable (skips existing).
    ``limit`` caps the number of chips (for a quick run). Returns
    ``{"s1": [...], "label": [...]}`` of local paths in matched order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"s1": [], "label": []}
    for kind, sub in [("s1", "S1Hand"), ("label", "LabelHand")]:
        dest = out_dir / sub
        dest.mkdir(parents=True, exist_ok=True)
        names = _list_objects(f"{_PREFIX}/{sub}/")
        if limit:
            names = names[:limit]
        log.info("Sen1Floods11 %s: %d chips", sub, len(names))
        for i, name in enumerate(names):
            fp = dest / Path(name).name
            if not fp.exists():
                urllib.request.urlretrieve(_DL.format(b=_BUCKET, name=urllib.parse.quote(name)), fp)
            result[kind].append(fp)
            if (i + 1) % 50 == 0:
                log.info("  %s %d/%d", sub, i + 1, len(names))
    log.info("Sen1Floods11 download complete: %d S1 / %d labels",
             len(result["s1"]), len(result["label"]))
    return result
