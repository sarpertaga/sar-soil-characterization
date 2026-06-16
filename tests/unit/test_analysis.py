import numpy as np

from soilgeo.analysis.classification import CLASS_NAMES, classify_surface_response
from soilgeo.analysis.statistics import (
    kruskal_wallis_test,
    spearman_correlation,
    stratify_by_quantiles,
)

# ── Statistics ────────────────────────────────────────────────────────────────

def test_stratify_returns_correct_labels():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    labels = stratify_by_quantiles(values, n_quantiles=4)
    assert labels.shape == values.shape
    assert set(labels) == {0, 1, 2, 3}


def test_kruskal_wallis_rejects_different_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 100)
    b = rng.normal(5, 1, 100)
    result = kruskal_wallis_test([a, b])
    assert result["p_value"] < 0.001


def test_kruskal_wallis_accepts_same_distribution():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 100)
    b = rng.normal(0, 1, 100)
    result = kruskal_wallis_test([a, b])
    assert result["p_value"] > 0.001


def test_spearman_perfect_correlation():
    x = np.arange(50, dtype=float)
    y = x * 2.0 + 1.0
    result = spearman_correlation(x, y)
    assert abs(result["rho"] - 1.0) < 1e-6


# ── Classification ────────────────────────────────────────────────────────────

def test_classify_output_shape():
    rng = np.random.default_rng(42)
    n = 1000
    features = {
        "moisture_index": rng.uniform(0, 1, n),
        "twi": rng.uniform(2, 15, n),
        "slope": rng.uniform(0, 30, n),
    }
    labels = classify_surface_response(features, n_clusters=5, random_state=0)
    assert labels.shape == (n,)
    assert labels.dtype == np.uint8


def test_classify_produces_n_classes():
    rng = np.random.default_rng(7)
    n = 2000
    features = {
        "moisture_index": rng.uniform(0, 1, n),
        "twi": rng.uniform(2, 15, n),
        "slope": rng.uniform(0, 30, n),
    }
    labels = classify_surface_response(features, n_clusters=5, random_state=0)
    assert len(np.unique(labels[labels != 255])) == 5


def test_class_names_has_five_entries():
    assert len(CLASS_NAMES) == 5
