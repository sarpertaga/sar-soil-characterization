"""
Climate data acquisition for V3 features (V3-F3, V3-F6).

* **ERA5-Land** (2 m temperature, potential evaporation) via the Copernicus
  Climate Data Store ``cdsapi`` — requires a CDS account and a ``~/.cdsapirc``
  (or ``CDSAPI_URL`` / ``CDSAPI_KEY`` env vars).
* **CHIRPS** v2.0 monthly precipitation via the UCSB Climate Hazards Center
  public HTTP archive — no credentials.

Downloaded grids feed :func:`soilgeo.features.climate.spi` (SPI-3) and the
climate normals used as auxiliary features. Not runnable in CI (network +
credentials).
"""
import urllib.request
from pathlib import Path

from soilgeo.utils.logging import get_logger

log = get_logger(__name__)

_CHIRPS_BASE = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs"
)


def download_chirps_monthly(
    year_start: int,
    year_end: int,
    output_dir: Path,
) -> list[Path]:
    """
    Download CHIRPS v2.0 global monthly precipitation GeoTIFFs (~5 km) for the
    given year range. Resumable: existing files are skipped. Caller clips to the
    AOI downstream. Returns the list of local paths successfully present.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            name = f"chirps-v2.0.{year}.{month:02d}.tif.gz"
            dest = output_dir / name
            if dest.exists():
                paths.append(dest)
                continue
            url = f"{_CHIRPS_BASE}/{name}"
            try:
                log.info("CHIRPS download: %s", name)
                urllib.request.urlretrieve(url, dest)
                paths.append(dest)
            except Exception as exc:  # noqa: BLE001
                log.warning("CHIRPS download failed for %s: %s", name, exc)
    return paths


def download_era5_land(
    bbox_wgs84: dict,
    variables: list[str],
    year_start: int,
    year_end: int,
    output_path: Path,
) -> Path:
    """
    Download monthly-aggregated ERA5-Land variables for the AOI via ``cdsapi``.
    Writes a single NetCDF covering the year range. Requires CDS credentials.
    """
    if output_path.exists():
        log.info("Skipping ERA5-Land (exists): %s", output_path.name)
        return output_path

    import cdsapi

    # CDS area order is [North, West, South, East]
    area = [bbox_wgs84["north"], bbox_wgs84["west"], bbox_wgs84["south"], bbox_wgs84["east"]]
    years = [str(y) for y in range(year_start, year_end + 1)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    log.info("ERA5-Land request: vars=%s years=%s area=%s", variables, years, area)
    client.retrieve(
        "reanalysis-era5-land-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": variables,
            "year": years,
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": "00:00",
            "area": area,
            "format": "netcdf",
        },
        str(output_path),
    )
    log.info("ERA5-Land saved: %s", output_path.name)
    return output_path
