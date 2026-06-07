import numpy as np

from soilgeo.features.timeseries import (
    compute_temporal_stats,
    compute_wet_dry_delta,
)

NODATA = -9999.0


def _stack():
    """[T=4, H=2, W=2] with a known per-pixel series and one all-NODATA pixel."""
    s = np.empty((4, 2, 2), dtype=np.float32)
    # pixel (0,0): 0,1,2,3
    s[:, 0, 0] = [0.0, 1.0, 2.0, 3.0]
    # pixel (0,1): constant 5
    s[:, 0, 1] = [5.0, 5.0, 5.0, 5.0]
    # pixel (1,0): has one NODATA -> uses remaining {10,20,30}
    s[:, 1, 0] = [10.0, NODATA, 20.0, 30.0]
    # pixel (1,1): all NODATA
    s[:, 1, 1] = [NODATA, NODATA, NODATA, NODATA]
    return s


def test_output_shapes_and_dtype():
    stats = compute_temporal_stats(_stack(), nodata=NODATA)
    for key in ("mean", "std", "p10", "p90", "amplitude"):
        assert stats[key].shape == (2, 2)
        assert stats[key].dtype == np.float32


def test_temporal_mean_ignores_nodata():
    stats = compute_temporal_stats(_stack(), nodata=NODATA)
    assert np.isclose(stats["mean"][0, 0], 1.5)        # mean(0,1,2,3)
    assert np.isclose(stats["mean"][0, 1], 5.0)        # constant
    assert np.isclose(stats["mean"][1, 0], 20.0)       # mean(10,20,30), nodata skipped


def test_constant_pixel_has_zero_std_and_amplitude():
    stats = compute_temporal_stats(_stack(), nodata=NODATA)
    assert np.isclose(stats["std"][0, 1], 0.0)
    assert np.isclose(stats["amplitude"][0, 1], 0.0)


def test_amplitude_is_p90_minus_p10():
    stats = compute_temporal_stats(_stack(), nodata=NODATA)
    valid = stats["amplitude"] != NODATA      # relationship defined on valid pixels only
    np.testing.assert_allclose(
        stats["amplitude"][valid], (stats["p90"] - stats["p10"])[valid], rtol=1e-5
    )


def test_all_nodata_pixel_stays_nodata():
    stats = compute_temporal_stats(_stack(), nodata=NODATA)
    for key in ("mean", "std", "p10", "p90", "amplitude"):
        assert stats[key][1, 1] == NODATA


def test_wet_dry_delta():
    # wet = dates {2,3} (values 2,3 -> 2.5), dry = dates {0,1} (0,1 -> 0.5) for px(0,0)
    delta = compute_wet_dry_delta(_stack(), wet_idx=[2, 3], dry_idx=[0, 1], nodata=NODATA)
    assert delta.shape == (2, 2)
    assert delta.dtype == np.float32
    assert np.isclose(delta[0, 0], 2.0)       # 2.5 - 0.5
    assert np.isclose(delta[0, 1], 0.0)       # constant
    assert delta[1, 1] == NODATA              # all-nodata pixel
