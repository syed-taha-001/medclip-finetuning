"""
test_nbr_empty_pairs.py

Deliberately calls neighbor_loss (L_nbr) with zero pairs against both
implementations. losses.py and metrics.py make OPPOSITE choices on
purpose:

  - losses.py (training-time): returns a 0.0-valued tensor. A batch that
    happens to sample zero neighbor pairs shouldn't crash the training
    loop -- it just contributes no geometry-preservation penalty that step.

  - metrics.py (evaluation-time): raises ValueError. A reported "0.0
    neighborhood distortion" with zero pairs actually compared would
    misrepresent the result as "geometry perfectly preserved" when in
    fact nothing was measured at all.

This test exists specifically to make sure that asymmetry is intentional
and doesn't silently drift out of sync between the two modules.
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
def dummy_embeddings():
    rng = np.random.default_rng(0)
    z_pre = rng.standard_normal((10, 4))
    z_post = rng.standard_normal((10, 4))
    return z_pre, z_post


@pytest.mark.parametrize("empty_pairs", [[], np.empty((0, 2), dtype=int), None])
def test_losses_returns_zero_on_empty_pairs(dummy_embeddings, empty_pairs):
    z_pre, z_post = dummy_embeddings
    z_pre_t = torch.tensor(z_pre, dtype=torch.float64)
    z_post_t = torch.tensor(z_post, dtype=torch.float64)

    result = losses.neighbor_loss(z_pre_t, z_post_t, empty_pairs)

    assert torch.is_tensor(result)
    assert result.item() == 0.0
    assert not torch.isnan(result).any()


def test_losses_empty_pairs_does_not_break_combined_loss(dummy_embeddings):
    """The whole point of the sane-default behavior: L_nbr contributing 0
    must not stop the rest of L_adapt (task + drift + cos) from working,
    and the combined loss must still be a valid, backprop-able scalar."""
    z_pre, z_post_leaf = dummy_embeddings
    z_pre_t = torch.tensor(z_pre, dtype=torch.float64)
    z_post_t = torch.tensor(z_post_leaf, dtype=torch.float64, requires_grad=True)
    task_loss = (z_post_t ** 2).sum()  # any differentiable stand-in

    total, components = losses.adaptation_loss(
        task_loss, z_pre_t, z_post_t, pairs=[], lambda1=1.0, lambda2=1.0, lambda3=1.0
    )
    assert components["nbr"] == 0.0
    assert not torch.isnan(total)
    total.backward()
    assert z_post_t.grad is not None
    assert not torch.isnan(z_post_t.grad).any()


@pytest.mark.parametrize("empty_pairs", [[], np.empty((0, 2), dtype=int), None])
def test_metrics_raises_value_error_on_empty_pairs(dummy_embeddings, empty_pairs):
    z_pre, z_post = dummy_embeddings
    with pytest.raises(ValueError, match="zero pairs"):
        metrics.neighbor_loss(z_pre, z_post, empty_pairs)


def test_metrics_error_message_is_actionable(dummy_embeddings):
    """Not just that it raises, but that the message actually tells you
    what likely went wrong (empty pair set P), not just a generic
    IndexError from indexing with an empty array."""
    z_pre, z_post = dummy_embeddings
    with pytest.raises(ValueError) as exc_info:
        metrics.neighbor_loss(z_pre, z_post, [])
    message = str(exc_info.value)
    assert "pair" in message.lower()
    assert "empty" in message.lower() or "zero" in message.lower()


def test_nonempty_pairs_unaffected_by_empty_path(dummy_embeddings):
    """Sanity guard: confirm the empty-pairs special-case doesn't
    accidentally short-circuit the normal non-empty path too."""
    z_pre, z_post = dummy_embeddings
    z_pre_t = torch.tensor(z_pre, dtype=torch.float64)
    z_post_t = torch.tensor(z_post, dtype=torch.float64)
    pairs = [[0, 1], [2, 3]]

    result = losses.neighbor_loss(z_pre_t, z_post_t, pairs)
    assert result.item() != 0.0  # random embeddings -> essentially never exactly 0

    # metrics.py should compute a real number, not raise, for non-empty pairs
    value = metrics.neighbor_loss(z_pre, z_post, pairs)
    assert isinstance(value, float)
