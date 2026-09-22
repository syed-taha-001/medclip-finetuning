"""
metrics.py

Pure-numpy, non-differentiable counterparts of losses.py's L_drift / L_cos /
L_nbr, used only AFTER training to measure and report numbers -- never
plugged into a training loop, so no autograd requirement. Also adds
per-sample drift and neighborhood stability (S_k), which aren't training
losses at all, just evaluation metrics (proposal Eqs. 10-11, adapted here
to compare the pre-adaptation vs. post-adaptation embedding of the same
clean sample, rather than clean-vs-attacked -- the attack-phase usage of
the same formulas comes later in the project).
"""

import numpy as np
from scipy.spatial.distance import cdist


def drift_loss(z_pre: np.ndarray, z_post: np.ndarray) -> float:
    """Eq 5 (numpy): mean squared L2 displacement between paired embeddings."""
    diff = z_post - z_pre
    return float(np.mean(np.sum(diff ** 2, axis=-1)))


def cosine_loss(z_pre: np.ndarray, z_post: np.ndarray, eps: float = 1e-8) -> float:
    """Eq 6 (numpy): mean(1 - cosine_similarity(z_post_i, z_pre_i))."""
    pre_norm = np.linalg.norm(z_pre, axis=-1)
    post_norm = np.linalg.norm(z_post, axis=-1)
    dot = np.sum(z_pre * z_post, axis=-1)
    cos_sim = dot / (pre_norm * post_norm + eps)
    return float(np.mean(1 - cos_sim))


def neighbor_loss(z_pre: np.ndarray, z_post: np.ndarray, pairs, weights=None,
                   distance: str = "euclidean") -> float:
    """Eq 7 (numpy): weighted mean squared change in pairwise distance.

    Empty-pairs behavior (evaluation-time default, distinct from
    losses.py): RAISES ValueError rather than returning 0.0. A reported
    "0.0 neighborhood distortion" with zero pairs actually compared would
    misrepresent the result -- it means nothing was measured, not that
    geometry was perfectly preserved. Training-time code (losses.py) makes
    the opposite choice deliberately; see that module's docstring.
    """
    if pairs is None or len(pairs) == 0:
        raise ValueError(
            "neighbor_loss (metrics.py) called with zero pairs -- cannot "
            "compute a neighborhood-distortion metric with nothing to "
            "compare. This likely means the clinically-informative pair "
            "set P was empty or wasn't passed correctly."
        )

    pairs = np.asarray(pairs)
    i_idx, j_idx = pairs[:, 0], pairs[:, 1]

    if weights is None:
        weights = np.ones(len(pairs))
    else:
        weights = np.asarray(weights)

    if distance == "euclidean":
        d_post = np.linalg.norm(z_post[i_idx] - z_post[j_idx], axis=-1)
        d_pre = np.linalg.norm(z_pre[i_idx] - z_pre[j_idx], axis=-1)
    elif distance == "cosine":
        def cos_dist(a, b):
            an = np.linalg.norm(a, axis=-1)
            bn = np.linalg.norm(b, axis=-1)
            return 1 - np.sum(a * b, axis=-1) / (an * bn + 1e-8)
        d_post = cos_dist(z_post[i_idx], z_post[j_idx])
        d_pre = cos_dist(z_pre[i_idx], z_pre[j_idx])
    else:
        raise ValueError(f"Unknown distance metric: {distance!r} (expected 'euclidean' or 'cosine')")

    sq_diff = (d_post - d_pre) ** 2
    return float(np.sum(weights * sq_diff) / len(pairs))


def per_sample_drift(z_pre: np.ndarray, z_post: np.ndarray) -> np.ndarray:
    """Per-sample Euclidean displacement ||z_post_i - z_pre_i||_2 (Eq 10
    adapted: pre- vs. post-adaptation on the same clean sample, rather than
    clean-vs-attacked). Feeds per_sample_drift.csv for plotting; drift_loss
    above is just the mean of this array, squared per-sample first.
    """
    return np.linalg.norm(z_post - z_pre, axis=-1)


def neighborhood_stability(z_pre: np.ndarray, z_post: np.ndarray, k: int) -> np.ndarray:
    """Eq 11 adapted: S_k(x_i) = |N_k(x_i) intersect N'_k(x_i)| / k, where
    N_k is the k-nearest-neighbor set in the pre-adaptation embedding space
    and N'_k the corresponding set in the post-adaptation space. Returns one
    S_k score per sample, in [0, 1]; 1.0 means the sample's local
    neighborhood was fully preserved by fine-tuning, 0.0 means fully
    disrupted.
    """
    n = z_pre.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be smaller than the number of samples ({n}).")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")

    def knn_indices(z):
        dists = cdist(z, z)
        np.fill_diagonal(dists, np.inf)  # exclude self as its own neighbor
        return np.argsort(dists, axis=1)[:, :k]

    nn_pre = knn_indices(z_pre)
    nn_post = knn_indices(z_post)

    scores = np.empty(n)
    for i in range(n):
        overlap = len(set(nn_pre[i].tolist()) & set(nn_post[i].tolist()))
        scores[i] = overlap / k
    return scores
