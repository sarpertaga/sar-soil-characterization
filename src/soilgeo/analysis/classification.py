"""Surface Response Classification via k-means on MI, TWI, slope."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from soilgeo.utils.logging import get_logger

log = get_logger(__name__)

# Labels assigned post-hoc by ascending wetness score of cluster centroids
CLASS_NAMES = {
    0: "dry",
    1: "moderately_dry",
    2: "transitional",
    3: "seasonally_wet",
    4: "persistently_wet",
}


def classify_surface_response(
    features: dict,
    n_clusters: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """
    K-means on [moisture_index, twi, slope].
    Returns uint8 label array (0=driest … n-1=wettest), 255=nodata.
    """
    keys = ["moisture_index", "twi", "slope"]
    X = np.column_stack([features[k] for k in keys])

    valid_mask = ~np.any(np.isnan(X) | (X == -9999.0), axis=1)
    X_valid = X[valid_mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    raw_labels = km.fit_predict(X_scaled)

    # Re-order clusters by wetness: MI↑ + TWI↑ − slope↑
    centroids = scaler.inverse_transform(km.cluster_centers_)
    wetness_score = centroids[:, 0] + 0.05 * centroids[:, 1] - 0.01 * centroids[:, 2]
    rank_order = np.argsort(wetness_score)
    remap = {old: new for new, old in enumerate(rank_order)}
    ordered_labels = np.vectorize(remap.get)(raw_labels).astype(np.uint8)

    output = np.full(len(features["moisture_index"]), 255, dtype=np.uint8)
    output[valid_mask] = ordered_labels

    log.info(
        "Surface Response Classes: %s",
        {CLASS_NAMES.get(k, k): int(np.sum(output == k)) for k in range(n_clusters)},
    )
    return output
