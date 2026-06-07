import numpy as np

from soilgeo.models.tiling import (
    assign_blocks,
    make_tile_index,
    split_blocks,
)


def test_tile_index_covers_raster():
    tiles = make_tile_index(H=500, W=600, tile=256, overlap=32)
    # every pixel must fall inside at least one tile
    covered = np.zeros((500, 600), dtype=bool)
    for r, c, h, w in tiles:
        covered[r:r + h, c:c + w] = True
    assert covered.all()


def test_tile_stride_and_edge_clamp():
    tiles = make_tile_index(H=500, W=500, tile=256, overlap=32)
    # stride = tile - overlap = 224
    row_offs = sorted({t[0] for t in tiles})
    assert row_offs[1] - row_offs[0] == 224
    # no tile exceeds the raster bounds
    for r, c, h, w in tiles:
        assert r + h <= 500
        assert c + w <= 500


def test_assign_blocks_groups_by_spatial_cell():
    tiles = make_tile_index(H=2000, W=2000, tile=256, overlap=32)
    block_ids = assign_blocks(tiles, block_km=5, pixel_m=10)  # 5 km / 10 m = 500 px blocks
    assert len(block_ids) == len(tiles)
    # tiles near origin share a block; far-apart tiles differ
    assert block_ids[0] != block_ids[-1]


def test_split_blocks_no_leakage():
    tiles = make_tile_index(H=3000, W=3000, tile=256, overlap=32)
    block_ids = assign_blocks(tiles, block_km=5, pixel_m=10)
    split = split_blocks(block_ids, ratios=(0.7, 0.15, 0.15), seed=42)
    # each split holds a set of block ids; no block id appears in two splits (V3-T1)
    train_b = {block_ids[i] for i in split["train"]}
    val_b = {block_ids[i] for i in split["val"]}
    test_b = {block_ids[i] for i in split["test"]}
    assert train_b.isdisjoint(val_b)
    assert train_b.isdisjoint(test_b)
    assert val_b.isdisjoint(test_b)
    # all tiles assigned exactly once
    total = len(split["train"]) + len(split["val"]) + len(split["test"])
    assert total == len(tiles)


def test_split_is_deterministic():
    tiles = make_tile_index(H=3000, W=3000, tile=256, overlap=32)
    block_ids = assign_blocks(tiles, block_km=5, pixel_m=10)
    a = split_blocks(block_ids, ratios=(0.7, 0.15, 0.15), seed=7)
    b = split_blocks(block_ids, ratios=(0.7, 0.15, 0.15), seed=7)
    assert a == b
