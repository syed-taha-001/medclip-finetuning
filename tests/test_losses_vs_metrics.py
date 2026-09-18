"""
test_losses_vs_metrics.py

Feeds identical synthetic data into losses.py (torch) and metrics.py (numpy)
and confirms they produce the same numbers. Catches translation bugs
between the two implementations of the same equations (Eqs. 5-7) -- one
differentiable and plugged into training, the other numpy-only and used
for post-hoc reporting.

No dataset, checkpoint, or GPU required -- pure synthetic tensors.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import losses  # noqa: E402
import metrics  # noqa: E402


@pytest.fixture
def synthetic_embeddings():
    rng = np.random.default_rng(0)
    N, d = 20, 8
    z_pre = rng.standard_normal((N, d))
    z_post = z_pre + 0.1 * rng.standard_normal((N, d))
    return z_pre.astype(np.float64), z_post.astype(np.float64)


@pytest.fixture
def synthetic_pairs():
    rng = np.random.default_rng(1)
    pairs = rng.integers(0, 20, size=(15, 2))
    weights = rng.uniform(0.5, 2.0, size=15)
    return pairs, weights


def to_torch(z):
    return torch.tensor(z, dtype=torch.float64)


def test_drift_loss_matches(synthetic_embeddings):
    z_pre, z_post = synthetic_embeddings
    t = losses.drift_loss(to_torch(z_pre), to_torch(z_post)).item()
    n = metrics.drift_loss(z_pre, z_post)
    assert np.isclose(t, n)


def test_cosine_loss_matches(synthetic_embeddings):
    z_pre, z_post = synthetic_embeddings
    t = losses.cosine_loss(to_torch(z_pre), to_torch(z_post)).item()
    n = metrics.cosine_loss(z_pre, z_post)
    assert np.isclose(t, n)


@pytest.mark.parametrize("distance", ["euclidean", "cosine"])
def test_neighbor_loss_matches(synthetic_embeddings, synthetic_pairs, distance):
    z_pre, z_post = synthetic_embeddings
    pairs, weights = synthetic_pairs
    t = losses.neighbor_loss(to_torch(z_pre), to_torch(z_post), pairs, weights, distance).item()
    n = metrics.neighbor_loss(z_pre, z_post, pairs, weights, distance)
    assert np.isclose(t, n)


def test_neighbor_loss_unweighted_matches(synthetic_embeddings, synthetic_pairs):
    """Confirms the default (weights=None -> all-ones) path agrees too,
    not just the explicitly-weighted path."""
    z_pre, z_post = synthetic_embeddings
    pairs, _ = synthetic_pairs
    t = losses.neighbor_loss(to_torch(z_pre), to_torch(z_post), pairs).item()
    n = metrics.neighbor_loss(z_pre, z_post, pairs)
    assert np.isclose(t, n)


def test_identical_embeddings_give_zero_loss():
    """No adaptation (z_post == z_pre) must give exactly zero for all three
    terms in both implementations -- a basic sanity anchor independent of
    torch/numpy agreement."""
    z = np.random.default_rng(2).standard_normal((10, 4))
    pairs = [[0, 1], [2, 3], [4, 5]]

    assert losses.drift_loss(to_torch(z), to_torch(z)).item() == 0.0
    assert metrics.drift_loss(z, z) == 0.0

    assert abs(losses.cosine_loss(to_torch(z), to_torch(z)).item()) < 1e-6
    assert abs(metrics.cosine_loss(z, z)) < 1e-6

    assert losses.neighbor_loss(to_torch(z), to_torch(z), pairs).item() == 0.0
    assert metrics.neighbor_loss(z, z, pairs) == 0.0


def test_pairs_as_plain_list_of_tuples():
    """pairs doesn't have to be a numpy array or torch tensor -- a plain
    Python list of (i, j) tuples must work too, since that's the more
    natural way pair sets get constructed upstream."""
    rng = np.random.default_rng(3)
    z_pre, z_post = rng.standard_normal((6, 3)), rng.standard_normal((6, 3))
    pairs_as_list = [(0, 1), (2, 3), (4, 5)]

    t = losses.neighbor_loss(to_torch(z_pre), to_torch(z_post), pairs_as_list).item()
    n = metrics.neighbor_loss(z_pre, z_post, pairs_as_list)
    assert np.isclose(t, n)


def test_neighborhood_stability_hand_verifiable_case():
    """Two tight 2D clusters, k=1. Moving one point into the other cluster
    should collapse S_1 to 0 for the affected points, and stay 1.0 when
    embeddings are unchanged."""
    z_pre = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [10.0, 10.0],
        [10.1, 10.0],
    ])
    s_identical = metrics.neighborhood_stability(z_pre, z_pre.copy(), k=1)
    assert np.allclose(s_identical, 1.0)

    z_post_disrupted = z_pre.copy()
    z_post_disrupted[0] = [10.05, 10.0]  # point A moved next to the C/D cluster
    s_disrupted = metrics.neighborhood_stability(z_pre, z_post_disrupted, k=1)
    assert s_disrupted[0] == 0.0
    assert np.all((s_disrupted >= 0.0) & (s_disrupted <= 1.0))


def test_neighborhood_stability_rejects_invalid_k():
    z = np.random.default_rng(4).standard_normal((4, 2))
    with pytest.raises(ValueError):
        metrics.neighborhood_stability(z, z, k=4)  # k must be < n
    with pytest.raises(ValueError):
        metrics.neighborhood_stability(z, z, k=0)  # k must be >= 1


def test_unknown_distance_metric_raises():
    z_pre, z_post = np.random.default_rng(5).standard_normal((6, 3)), np.random.default_rng(6).standard_normal((6, 3))
    pairs = [[0, 1], [2, 3]]
    with pytest.raises(ValueError):
        metrics.neighbor_loss(z_pre, z_post, pairs, distance="manhattan")
    with pytest.raises(ValueError):
        losses.neighbor_loss(to_torch(z_pre), to_torch(z_post), pairs, distance="manhattan")
