"""
Multi-date Sentinel-1 time-series acquisition (V3-F6).

Builds a regular calendar of acquisition dates over the configured window and
fetches one VV+VH σ⁰ composite per date via the existing CDSE Sentinel Hub
client. Each date is written as its own 2-band COG and the fetch is
**resumable** — already-present dates are skipped — because a full year of S1
is dozens of requests.

To keep incidence-angle geometry consistent (DR-R2), pass
``orbit_direction`` so every request filters to a single pass direction; the
per-date median over a narrow window then approximates a single relative orbit
when ``step_days`` matches the revisit cycle.

Requires Sentinel Hub credentials (``SH_CLIENT_ID`` / ``SH_CLIENT_SECRET``);
not runnable in CI.
"""
from datetime import date, datetime, timedelta
from pathlib import Path

from soilgeo.acquisition.sentinel_hub import fetch_backscatter
from soilgeo.utils.logging import get_logger

log = get_logger(__name__)


def build_date_windows(
    start: str,
    end: str,
    step_days: int = 12,
) -> list[tuple[str, str]]:
    """
    Build ``(window_start, window_end)`` intervals spaced ``step_days`` apart
    over ``[start, end]``. Each window spans ``step_days`` so every acquisition
    in the revisit cycle contributes to that date's median composite.
    """
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date()
    windows = []
    cur = d0
    while cur <= d1:
        w_end = min(cur + timedelta(days=step_days - 1), d1)
        windows.append((cur.isoformat(), w_end.isoformat()))
        cur += timedelta(days=step_days)
    return windows


def date_to_season_index(
    windows: list[tuple[str, str]],
    wet_months: list[int],
    dry_months: list[int],
) -> tuple[list[int], list[int]]:
    """
    Classify each window into wet / dry season by the month of its start date.
    Returns ``(wet_idx, dry_idx)`` lists of positions into ``windows`` — used by
    :func:`soilgeo.features.timeseries.compute_wet_dry_delta`.
    """
    wet_idx, dry_idx = [], []
    for i, (w_start, _w_end) in enumerate(windows):
        m = date.fromisoformat(w_start).month
        if m in wet_months:
            wet_idx.append(i)
        elif m in dry_months:
            dry_idx.append(i)
    return wet_idx, dry_idx


def fetch_s1_timeseries(
    bbox_wgs84: dict,
    start: str,
    end: str,
    output_dir: Path,
    aoi_name: str,
    step_days: int = 12,
    resolution_m: int = 10,
    orbit_direction: str | None = None,
) -> list[Path]:
    """
    Fetch a resumable S1 VV+VH time series. Returns the ordered list of per-date
    COG paths (band1=VV_dB, band2=VH_dB). Missing/failed dates are skipped with
    a warning so a partial series can still be assembled.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    windows = build_date_windows(start, end, step_days)
    log.info("S1 time series: %d windows %s→%s (step=%dd)", len(windows), start, end, step_days)

    paths = []
    for w_start, w_end in windows:
        out = output_dir / f"s1_{aoi_name}_{w_start}.tif"
        try:
            fetch_backscatter(
                bbox_wgs84=bbox_wgs84,
                time_interval=(w_start, w_end),
                output_path=out,
                resolution_m=resolution_m,
                median_composite=True,
                orbit_direction=orbit_direction,
            )
            paths.append(out)
        except Exception as exc:  # noqa: BLE001 — keep series resumable on transient errors
            log.warning("S1 fetch failed for %s: %s", w_start, exc)
    return paths
