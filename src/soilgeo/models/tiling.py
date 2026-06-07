"""
Tile inventory and spatial-block train/val/test splitting (V3-T1).

Deep-learning samples are 256×256 px tiles with 32 px overlap. To prevent
spatial-autocorrelation leakage (risk R4), tiles are grouped into spatial
blocks (default 5 km) and whole blocks are assigned to a single split — a tile
from one block never spans more than one of train/val/test.
"""
import numpy as np

from soilgeo.utils.logging import get_logger

log = get_logger(__name__)

Tile = tuple[int, int, int, int]  # (row_off, col_off, height, width)


def make_tile_index(H: int, W: int, tile: int = 256, overlap: int = 32) -> list[Tile]:
    """
    Generate tile windows covering an ``H×W`` raster with the given tile size
    and overlap. Tiles use a stride of ``tile - overlap``; the final tile in
    each row/column is clamped to the raster edge so coverage is complete and
    no tile exceeds the bounds.
    """
    if overlap >= tile:
        raise ValueError("overlap must be smaller than tile size")
    stride = tile - overlap

    def _offsets(extent: int) -> list[int]:
        if extent <= tile:
            return [0]
        offs = list(range(0, extent - tile + 1, stride))
        if offs[-1] != extent - tile:
            offs.append(extent - tile)   # clamp final tile to the edge
        return offs

    row_offs = _offsets(H)
    col_offs = _offsets(W)
    tiles = [
        (r, c, min(tile, H - r), min(tile, W - c))
        for r in row_offs
        for c in col_offs
    ]
    log.info("Tile index: %d tiles (%d px, overlap=%d) over %dx%d", len(tiles), tile, overlap, H, W)
    return tiles


def assign_blocks(tiles: list[Tile], block_km: float, pixel_m: float) -> list[int]:
    """
    Map each tile to a spatial-block id based on the block-row/col of its
    top-left corner. Block size is ``block_km`` (converted to pixels via
    ``pixel_m``). Returns one integer block id per tile.
    """
    block_px = max(1, int(round(block_km * 1000.0 / pixel_m)))
    # Use a large multiplier on the block-row so (row, col) maps to a unique id.
    n_block_cols = 0
    ids = []
    for r, c, _h, _w in tiles:
        br, bc = r // block_px, c // block_px
        n_block_cols = max(n_block_cols, bc + 1)
        ids.append((br, bc))
    # flatten (br, bc) -> single int id
    return [br * (n_block_cols + 1) + bc for br, bc in ids]


def split_blocks(
    block_ids: list[int],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[int]]:
    """
    Split tiles into train/val/test by whole spatial blocks. Returns a dict
    mapping each split name to the list of **tile indices** (into ``block_ids``)
    belonging to it. Deterministic for a fixed seed; no block id is shared
    across splits (V3-T1 invariant).
    """
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    unique_blocks = sorted(set(block_ids))
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(unique_blocks))

    n = len(shuffled)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    train_b = set(shuffled[:n_train])
    val_b = set(shuffled[n_train:n_train + n_val])
    # remaining blocks fall through to the test split below

    split: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for i, b in enumerate(block_ids):
        if b in train_b:
            split["train"].append(i)
        elif b in val_b:
            split["val"].append(i)
        else:
            split["test"].append(i)

    log.info(
        "Spatial-block split: %d blocks -> train=%d val=%d test=%d tiles",
        n, len(split["train"]), len(split["val"]), len(split["test"]),
    )
    return split
