"""
losses.py

Torch-native, differentiable implementations of the stability-constrained
fine-tuning loss terms (proposal Eqs. 5-8). These are the versions actually
plugged into finetune_medclip.py's training loop -- PyTorch needs to
backpropagate through them to update the model's weights.

See metrics.py for the numpy (non-differentiable) counterparts used only
for post-hoc evaluation and reporting.
"""

import torch
import torch.nn.functional as F


def drift_loss(z_pre: torch.Tensor, z_post: torch.Tensor) -> torch.Tensor:
    """Eq 5: L_drift = (1/N) * sum_i ||z_post_i - z_pre_i||_2^2

    z_pre, z_post: (N, d) tensors. z_pre should typically be .detach()'d
    by the caller (it's the frozen base model's output).
    """
    diff = z_post - z_pre
    per_sample_sq_norm = torch.sum(diff ** 2, dim=-1)
    return per_sample_sq_norm.mean()


def cosine_loss(z_pre: torch.Tensor, z_post: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Eq 6: L_cos = (1/N) * sum_i (1 - <z_post_i, z_pre_i> / (||z_post_i|| * ||z_pre_i||))"""
    cos_sim = F.cosine_similarity(z_post, z_pre, dim=-1, eps=eps)
    return (1 - cos_sim).mean()


def neighbor_loss(z_pre: torch.Tensor, z_post: torch.Tensor, pairs, weights=None,
                   distance: str = "euclidean") -> torch.Tensor:
    """Eq 7: L_nbr = (1/|P|) * sum_(i,j) w_ij * (d(z_post_i,z_post_j) - d(z_pre_i,z_pre_j))^2

    pairs: (M, 2) LongTensor/array/list of index pairs (i, j) into z_pre/z_post,
           or empty/None.
    weights: optional (M,) tensor of w_ij; defaults to all-ones.
    distance: "euclidean" or "cosine".

    Empty-pairs behavior (training-time default, distinct from metrics.py):
    returns a 0-valued scalar tensor rather than raising. A batch with no
    sampled neighbor pairs should not crash the training loop -- it simply
    contributes no geometry-preservation penalty for that step.
    """
    if pairs is None or len(pairs) == 0:
        return torch.zeros((), device=z_pre.device, dtype=z_pre.dtype)

    if not torch.is_tensor(pairs):
        pairs = torch.as_tensor(pairs, device=z_pre.device, dtype=torch.long)
    i_idx, j_idx = pairs[:, 0], pairs[:, 1]

    if weights is None:
        weights = torch.ones(pairs.shape[0], device=z_pre.device, dtype=z_pre.dtype)
    elif not torch.is_tensor(weights):
        weights = torch.as_tensor(weights, device=z_pre.device, dtype=z_pre.dtype)

    if distance == "euclidean":
        d_post = torch.norm(z_post[i_idx] - z_post[j_idx], dim=-1)
        d_pre = torch.norm(z_pre[i_idx] - z_pre[j_idx], dim=-1)
    elif distance == "cosine":
        d_post = 1 - F.cosine_similarity(z_post[i_idx], z_post[j_idx], dim=-1)
        d_pre = 1 - F.cosine_similarity(z_pre[i_idx], z_pre[j_idx], dim=-1)
    else:
        raise ValueError(f"Unknown distance metric: {distance!r} (expected 'euclidean' or 'cosine')")

    sq_diff = (d_post - d_pre) ** 2
    return (weights * sq_diff).sum() / pairs.shape[0]


def adaptation_loss(task_loss: torch.Tensor, z_pre: torch.Tensor, z_post: torch.Tensor,
                     pairs=None, weights=None,
                     lambda1: float = 1.0, lambda2: float = 1.0, lambda3: float = 1.0,
                     distance: str = "euclidean"):
    """Eq 8: L_adapt = L_task + lambda1*L_drift + lambda2*L_cos + lambda3*L_nbr

    Returns (total_loss_tensor, components_dict) where components_dict holds
    plain-float values of each term for logging to runs_log.csv.
    """
    l_drift = drift_loss(z_pre, z_post)
    l_cos = cosine_loss(z_pre, z_post)
    l_nbr = neighbor_loss(z_pre, z_post, pairs, weights, distance)

    total = task_loss + lambda1 * l_drift + lambda2 * l_cos + lambda3 * l_nbr

    components = {
        "task": float(task_loss.detach().item()) if torch.is_tensor(task_loss) else float(task_loss),
        "drift": float(l_drift.detach().item()),
        "cos": float(l_cos.detach().item()),
        "nbr": float(l_nbr.detach().item()),
        "total": float(total.detach().item()),
    }
    return total, components
