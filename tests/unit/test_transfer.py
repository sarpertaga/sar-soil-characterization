import numpy as np
import pytest

from soilgeo.analysis.transfer import align_features, common_features, cross_region_transfer

pytest.importorskip("lightgbm")


def test_common_features_is_sorted_intersection():
    a = ["slope", "twi", "curvature", "only_a"]
    b = ["twi", "slope", "only_b", "curvature"]
    assert common_features(a, b) == ["curvature", "slope", "twi"]


def test_align_features_reorders_columns():
    X = np.array([[1.0, 2.0, 3.0]])
    names = ["a", "b", "c"]
    out = align_features(X, names, ["c", "a"])
    assert out.tolist() == [[3.0, 1.0]]


def test_align_features_missing_raises():
    with pytest.raises(KeyError):
        align_features(np.zeros((1, 2)), ["a", "b"], ["a", "missing"])


def test_cross_region_transfer_recovers_linear_signal():
    rng = np.random.default_rng(0)
    # Shared generative rule across "regions": y = 3*f0 - 2*f1
    Xtr = rng.uniform(0, 1, (2000, 3))
    ytr = 3 * Xtr[:, 0] - 2 * Xtr[:, 1] + rng.normal(0, 0.01, 2000)
    Xte = rng.uniform(0, 1, (800, 3))
    yte = 3 * Xte[:, 0] - 2 * Xte[:, 1]
    names = ["f0", "f1", "f2"]
    res = cross_region_transfer(
        Xtr, ytr, names, Xte, yte, names,
        target="clay", train_region="A", test_region="B",
        n_estimators=100, max_train_rows=None,
    )
    assert res["n_common_features"] == 3
    assert res["out_of_region"]["r2"] > 0.9  # same rule -> transfers well


def test_cross_region_transfer_fails_on_region_specific_signal():
    rng = np.random.default_rng(1)
    Xtr = rng.uniform(0, 1, (2000, 2))
    ytr = 5 * Xtr[:, 0]                       # region A: depends on f0
    Xte = rng.uniform(0, 1, (800, 2))
    yte = 5 * Xte[:, 1]                       # region B: depends on f1 instead
    names = ["f0", "f1"]
    res = cross_region_transfer(
        Xtr, ytr, names, Xte, yte, names,
        target="clay", train_region="A", test_region="B",
        n_estimators=100, max_train_rows=None,
    )
    assert res["out_of_region"]["r2"] < 0.0   # opposite rule -> no transfer
