import numpy as np

from soilgeo.features.climate import rolling_sum, spi


def test_rolling_sum_values_and_length():
    s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = rolling_sum(s, window=3)
    assert out.shape == s.shape
    # first two are NaN (insufficient history), then 1+2+3, 2+3+4, 3+4+5
    assert np.isnan(out[0]) and np.isnan(out[1])
    np.testing.assert_allclose(out[2:], [6.0, 9.0, 12.0])


def test_spi_is_standardized():
    rng = np.random.default_rng(0)
    # long gamma-distributed monthly precip series
    precip = rng.gamma(shape=2.0, scale=20.0, size=600).astype(np.float64)
    z = spi(precip, scale=3)
    valid = z[~np.isnan(z)]
    assert abs(valid.mean()) < 0.15      # ~zero mean
    assert abs(valid.std() - 1.0) < 0.15  # ~unit std


def test_spi_sign_tracks_wet_and_dry():
    rng = np.random.default_rng(1)
    precip = rng.gamma(shape=2.0, scale=20.0, size=600)
    # force a wet run and a dry run at known positions
    precip[300:303] = 500.0   # very wet quarter
    precip[400:403] = 0.1     # drought quarter
    z = spi(precip, scale=3)
    assert z[302] > 1.0       # wet -> strongly positive
    assert z[402] < -1.0      # dry -> strongly negative


def test_spi_handles_zeros():
    rng = np.random.default_rng(2)
    precip = rng.gamma(shape=2.0, scale=20.0, size=600)
    precip[::5] = 0.0          # 20% exact zeros (arid regime)
    z = spi(precip, scale=3)
    valid = z[~np.isnan(z)]
    assert np.all(np.isfinite(valid))
