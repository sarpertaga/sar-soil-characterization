import numpy as np
import rasterio
from rasterio.transform import from_bounds

from soilgeo.models.cube import assemble_cube, build_matrix

NODATA = -9999.0


def _write(path, arr, bbox=(33.0, 37.0, 33.1, 37.1)):
    h, w = arr.shape
    transform = from_bounds(*bbox, w, h)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=NODATA,
    ) as dst:
        dst.write(arr.astype(np.float32), 1)
    return path


def test_assemble_cube_stacks_and_aligns(tmp_path):
    ref = _write(tmp_path / "ref.tif", np.ones((20, 20), np.float32))
    f1 = _write(tmp_path / "f1.tif", np.full((20, 20), 2.0, np.float32))
    f2 = _write(tmp_path / "f2.tif", np.full((10, 10), 3.0, np.float32))  # coarser → resampled
    cube, names, profile = assemble_cube(
        {"a": ref, "b": f1, "c": f2}, ref_path=ref, work_dir=tmp_path / "w"
    )
    assert cube.shape == (3, 20, 20)
    assert names == ["a", "b", "c"]          # sorted, deterministic
    assert np.isclose(cube[1].mean(), 2.0)
    assert np.isclose(cube[2].mean(), 3.0)   # resampled coarse band


def test_build_matrix_valid_pixels_and_groups(tmp_path):
    feat = np.full((20, 20), 5.0, np.float32)
    feat[0, 0] = NODATA                       # one invalid feature pixel
    ref = _write(tmp_path / "ref.tif", feat)
    cube, _names, _p = assemble_cube({"f": ref}, ref_path=ref, work_dir=tmp_path / "w")

    label = _write(tmp_path / "clay.tif", np.full((20, 20), 30.0, np.float32))
    X, y, groups, valid = build_matrix(
        cube, {"clay": label}, ref_path=ref, work_dir=tmp_path / "w", block_px=10
    )
    assert X.shape == (399, 1)                # 400 - 1 nodata pixel
    assert valid.sum() == 399
    assert set(y.keys()) == {"clay"}
    assert y["clay"].shape == (399,)
    # 20x20 grid, 10px blocks → up to 4 spatial blocks
    assert 1 <= len(np.unique(groups)) <= 4
