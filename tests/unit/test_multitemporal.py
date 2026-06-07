import numpy as np

from soilgeo.sar.multitemporal import (
    db_to_linear,
    linear_to_db,
    quegan_yu_filter,
)

NODATA = -9999.0


def test_db_linear_roundtrip():
    db = np.array([[-10.0, -20.0], [-5.0, -15.0]], dtype=np.float32)
    back = linear_to_db(db_to_linear(db, NODATA), NODATA)
    np.testing.assert_allclose(back, db, atol=1e-4)


def test_db_linear_preserves_nodata():
    db = np.array([[NODATA, -20.0]], dtype=np.float32)
    lin = db_to_linear(db, NODATA)
    assert lin[0, 0] == NODATA


def test_output_shape_matches_input():
    rng = np.random.default_rng(0)
    stack = rng.gamma(2.0, 1.0, size=(5, 16, 16)).astype(np.float32)
    out = quegan_yu_filter(stack, window=3, nodata=NODATA)
    assert out.shape == stack.shape
    assert out.dtype == np.float32


def test_constant_stack_unchanged():
    # Flat intensity everywhere -> local mean == value -> ratio == 1 -> output == input
    stack = np.full((4, 8, 8), 2.5, dtype=np.float32)
    out = quegan_yu_filter(stack, window=3, nodata=NODATA)
    np.testing.assert_allclose(out, stack, atol=1e-4)


def test_reduces_speckle_variance():
    # Multiplicative speckle on a constant scene; filtered stack should be smoother.
    rng = np.random.default_rng(42)
    truth = 3.0
    speckle = rng.gamma(shape=1.0, scale=truth, size=(8, 32, 32)).astype(np.float32)
    out = quegan_yu_filter(speckle, window=7, nodata=NODATA)
    assert out.var() < speckle.var()


def test_nodata_propagates():
    stack = np.full((3, 4, 4), 2.0, dtype=np.float32)
    stack[:, 0, 0] = NODATA
    out = quegan_yu_filter(stack, window=3, nodata=NODATA)
    assert np.all(out[:, 0, 0] == NODATA)
