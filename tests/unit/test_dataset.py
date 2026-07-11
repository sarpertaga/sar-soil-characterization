import numpy as np
import pytest

pytest.importorskip("torch")  # soilgeo-dl env only

from soilgeo.models.dataset import (  # noqa: E402
    NormStats,
    SoilTileDataset,
    apply_augmentation,
    compute_norm_stats,
)


def _cube(seed=0):
    """[C=3, H, W] synthetic feature cube with distinct per-channel scales."""
    rng = np.random.default_rng(seed)
    c0 = rng.normal(10.0, 2.0, (32, 32))
    c1 = rng.normal(-5.0, 0.5, (32, 32))
    c2 = rng.normal(0.0, 100.0, (32, 32))
    return np.stack([c0, c1, c2]).astype(np.float32)


def test_norm_stats_per_channel_from_training_only():
    train = [_cube(1), _cube(2), _cube(3)]
    stats = compute_norm_stats(train, nodata=-9999.0)
    assert isinstance(stats, NormStats)
    assert stats.mean.shape == (3,)
    assert stats.std.shape == (3,)
    # channel 2 has much larger spread than channel 1
    assert stats.std[2] > stats.std[1]


def test_norm_stats_ignore_nodata():
    cube = _cube(1)
    cube[0, :5, :5] = -9999.0
    stats = compute_norm_stats([cube], nodata=-9999.0)
    # mean of channel 0 stays near 10 despite the injected nodata block
    assert abs(stats.mean[0] - 10.0) < 1.5


def test_normalize_then_roundtrip():
    train = [_cube(1), _cube(2)]
    stats = compute_norm_stats(train, nodata=-9999.0)
    x = _cube(5)
    z = stats.normalize(x)
    # standardized channels have ~zero mean
    assert abs(float(z[1].mean())) < 0.5


def test_augmentation_geometric_only_preserves_shape_and_values():
    import torch
    x = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    y = torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4)
    for op in ("hflip", "vflip", "rot90"):
        xa, ya = apply_augmentation(x, y, op)
        assert xa.shape == x.shape
        assert ya.shape == y.shape
        # geometric ops only permute pixels — value multiset is unchanged
        assert torch.equal(torch.sort(xa.flatten())[0], torch.sort(x.flatten())[0])


def test_augmentation_applies_same_transform_to_x_and_y():
    import torch
    x = torch.zeros(1, 4, 4)
    x[0, 0, 0] = 1.0          # top-left marker
    y = torch.zeros(4, 4)
    y[0, 0] = 1.0
    xa, ya = apply_augmentation(x, y, "hflip")
    # marker moves to top-right in both feature and label
    assert xa[0, 0, -1] == 1.0
    assert ya[0, -1] == 1.0


def test_tile_dataset_regression_item_shapes():
    cube = np.stack([_cube(1)[0], _cube(2)[0]]).astype(np.float32)  # [2, 32, 32]
    label = np.full((32, 32), 30.0, np.float32)
    stats = compute_norm_stats([cube], nodata=-9999.0)
    tiles = [(0, 0, 16, 16), (0, 16, 16, 16)]
    ds = SoilTileDataset(cube, label, tiles, stats, tile_size=16, task="regression")
    assert len(ds) == 2
    x, y = ds[0]
    assert tuple(x.shape) == (2, 16, 16)
    assert tuple(y.shape) == (1, 16, 16)


def test_tile_dataset_pads_edge_tiles():
    cube = np.stack([_cube(1)[0]]).astype(np.float32)               # [1, 32, 32]
    label = np.full((32, 32), 5.0, np.float32)
    stats = compute_norm_stats([cube], nodata=-9999.0)
    tiles = [(24, 24, 8, 8)]                                        # 8x8 edge tile
    ds = SoilTileDataset(cube, label, tiles, stats, tile_size=16, task="regression")
    x, y = ds[0]
    assert tuple(x.shape) == (1, 16, 16)                           # padded up to 16
    assert tuple(y.shape) == (1, 16, 16)


def test_tile_dataset_never_feeds_nodata_to_the_network():
    """Invalid pixels and edge padding must reach the model as 0, not -9999."""
    import torch
    cube = np.stack([_cube(1)[0]]).astype(np.float32)               # [1, 32, 32]
    cube[0, :8, :8] = -9999.0                                       # nodata block
    label = np.full((32, 32), 5.0, np.float32)
    stats = compute_norm_stats([cube], nodata=-9999.0)
    tiles = [(0, 0, 16, 16), (24, 24, 8, 8)]                        # interior + edge tile
    ds = SoilTileDataset(cube, label, tiles, stats, tile_size=16, task="regression")
    for i in range(len(ds)):
        x, y = ds[i]
        assert torch.all(torch.abs(x) < 100.0), "raw NODATA leaked into features"
    # nodata feature pixels are exactly the neutral value 0
    x0, _ = ds[0]
    assert torch.all(x0[0, :8, :8] == 0.0)
    # label padding stays NODATA so the loss mask still excludes it
    _, y1 = ds[1]
    assert torch.all(y1[0, 8:, 8:] == -9999.0)
