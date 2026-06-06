import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path


@pytest.fixture
def synthetic_10m_raster(tmp_path) -> Path:
    path = tmp_path / "synthetic_vv.tif"
    data = np.random.default_rng(42).uniform(-20, 0, (200, 200)).astype(np.float32)
    transform = from_bounds(500000, 4160000, 502000, 4162000, 200, 200)
    with rasterio.open(
        path, "w",
        driver="GTiff", height=200, width=200, count=1,
        dtype="float32", crs="EPSG:32636",
        transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture
def synthetic_dem_raster(tmp_path) -> Path:
    path = tmp_path / "dem.tif"
    rows, cols = 200, 200
    elev = np.linspace(1000, 1100, rows * cols).reshape(rows, cols).astype(np.float32)
    transform = from_bounds(500000, 4160000, 502000, 4162000, cols, rows)
    with rasterio.open(
        path, "w",
        driver="GTiff", height=rows, width=cols, count=1,
        dtype="float32", crs="EPSG:32636",
        transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(elev, 1)
    return path
