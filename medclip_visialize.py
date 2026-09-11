"""
visualize_embeddings.py

Visualizes MedCLIP embedding space: normalization check, PCA variance,
UMAP projection, class separability, and (once z_post exists) pre/post
fine-tuning drift: joint UMAP + Procrustes alignment, trajectory arrows,
class centroid shift, and k-NN neighborhood overlap (ties to eq. 11, S_k).

Usage:
    python visualize_embeddings.py
    python visualize_embeddings.py --emb_dir ~/medclip_embeddings --k 10

Expects (produced by extract_medclip_embeddings.py):
    z_pre_embeddings.npy, z_pre_labels.npy, z_pre_paths.npy
Optionally, once you've fine-tuned and re-extracted:
    z_post_embeddings.npy, z_post_labels.npy, z_post_paths.npy
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import procrustes

try:
    import umap
    HAVE_UMAP = True
except ImportError:
    HAVE_UMAP = False
    print("umap-learn not installed — falling back to PCA for 2D projection.")
    print("Install with: pip install umap-learn --break-system-packages\n")


def load_split(emb_dir, name):
    """Load one embedding split (pre or post). Returns None if files missing."""
    emb_path = os.path.join(emb_dir, f"{name}_embeddings.npy")
    lbl_path = os.path.join(emb_dir, f"{name}_labels.npy")
    path_path = os.path.join(emb_dir, f"{name}_paths.npy")
    if not os.path.exists(emb_path):
        return None
    embeddings = np.load(emb_path)
    labels = np.load(lbl_path)
    paths = np.load(path_path)
    return {"embeddings": embeddings, "labels": labels, "paths": paths}


def check_normalization(embeddings, name):
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"[{name}] norm mean={norms.mean():.4f}  std={norms.std():.4f}  "
          f"(≈1.0 means already L2-normalized)")
    return norms


def plot_pca_variance(embeddings, out_path, name):
    pca = PCA(n_components=min(50, embeddings.shape[0] - 1))
    pca.fit(embeddings)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(cumvar) + 1), cumvar, marker="o", markersize=3)
    plt.axhline(0.9, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("number of PCA components")
    plt.ylabel("cumulative explained variance")
    plt.title(f"PCA variance — {name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    n90 = np.argmax(cumvar >= 0.9) + 1
    print(f"[{name}] components needed for 90% variance: {n90}")


def class_separability(embeddings, labels, name):
    classes = np.unique(labels)
    sims = cosine_similarity(embeddings)
    within, across = [], []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            (within if labels[i] == labels[j] else across).append(sims[i, j])
    print(f"[{name}] mean within-class cosine sim: {np.mean(within):.4f}")
    print(f"[{name}] mean across-class cosine sim: {np.mean(across):.4f}")


def project_2d(embeddings, n_neighbors=15):
    if HAVE_UMAP:
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1,
                             random_state=42)
        return reducer.fit_transform(embeddings)
    else:
        return PCA(n_components=2, random_state=42).fit_transform(embeddings)


def plot_single_projection(coords, labels, out_path, name):
    plt.figure(figsize=(6, 5))
    classes = np.unique(labels)
    colors = {"NORMAL": "#378ADD", "PNEUMONIA": "#D85A30"}
    for cls in classes:
        mask = labels == cls
        plt.scatter(coords[mask, 0], coords[mask, 1],
                    label=cls, alpha=0.7, s=25,
                    c=colors.get(cls, None))
    plt.legend()
    plt.title(f"Embedding space projection — {name}")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def compute_drift_metrics(pre, post):
    """Per-sample L2 drift (eq. 5 style) and cosine alignment (eq. 6 style).
    Assumes pre['paths'] and post['paths'] are index-aligned."""
    assert np.array_equal(pre["paths"], post["paths"]), \
        "z_pre and z_post paths are not aligned by index — check extraction order."

    z_pre, z_post = pre["embeddings"], post["embeddings"]
    l2_drift = np.linalg.norm(z_post - z_pre, axis=1)
    cos_sim = np.sum(z_pre * z_post, axis=1) / (
        np.linalg.norm(z_pre, axis=1) * np.linalg.norm(z_post, axis=1) + 1e-8
    )
    print(f"\n[drift] mean L2 drift:      {l2_drift.mean():.4f} (std {l2_drift.std():.4f})")
    print(f"[drift] mean cosine sim:    {cos_sim.mean():.4f} (1.0 = no directional change)")
    return l2_drift, cos_sim


def knn_overlap(pre, post, k=10):
    """Neighborhood-preservation check (S_k, eq. 11) in the ORIGINAL
    high-dim space — never on a 2D projection."""
    z_pre, z_post = pre["embeddings"], post["embeddings"]
    n = z_pre.shape[0]
    k = min(k, n - 1)

    nn_pre = NearestNeighbors(n_neighbors=k + 1).fit(z_pre)
    nn_post = NearestNeighbors(n_neighbors=k + 1).fit(z_post)
    _, idx_pre = nn_pre.kneighbors(z_pre)
    _, idx_post = nn_post.kneighbors(z_post)

    overlaps = []
    for i in range(n):
        set_pre = set(idx_pre[i][1:])   # drop self (index 0)
        set_post = set(idx_post[i][1:])
        overlaps.append(len(set_pre & set_post) / k)
    overlaps = np.array(overlaps)
    print(f"[S_k, k={k}] mean neighborhood overlap: {overlaps.mean():.4f} "
          f"(1.0 = perfectly preserved, 0.0 = fully scrambled)")
    return overlaps


def plot_joint_drift(pre, post, l2_drift, out_path, k_overlap=None):
    """Joint UMAP fit on concatenated pre+post, Procrustes-aligned,
    trajectory arrows colored by drift magnitude."""
    combined = np.vstack([pre["embeddings"], post["embeddings"]])
    coords = project_2d(combined)
    n = pre["embeddings"].shape[0]
    coords_pre, coords_post = coords[:n], coords[n:]

    # Procrustes: align post onto pre (rotation/scale only, no distortion)
    _, coords_post_aligned, _ = procrustes(coords_pre, coords_post)
    # rescale aligned coords back to a comparable visual range
    coords_post_aligned = coords_post_aligned * np.std(coords_pre) + np.mean(coords_pre, axis=0)

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(coords_pre[:, 0], coords_pre[:, 1], c="#888780",
                      s=20, alpha=0.6, label="z_pre")
    plt.scatter(coords_post_aligned[:, 0], coords_post_aligned[:, 1],
                c="#D85A30", s=20, alpha=0.6, label="z_post")

    norm_drift = (l2_drift - l2_drift.min()) / (l2_drift.max() - l2_drift.min() + 1e-8)
    for i in range(n):
        plt.plot([coords_pre[i, 0], coords_post_aligned[i, 0]],
                  [coords_pre[i, 1], coords_post_aligned[i, 1]],
                  color=plt.cm.viridis(norm_drift[i]), alpha=0.5, linewidth=0.8)

    plt.legend()
    plt.title("Embedding drift: z_pre → z_post (Procrustes-aligned)")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_centroid_shift(pre, post, out_path):
    classes = np.unique(pre["labels"])
    print("\n[centroids]")
    for cls in classes:
        c_pre = pre["embeddings"][pre["labels"] == cls].mean(axis=0)
        c_post = post["embeddings"][post["labels"] == cls].mean(axis=0)
        shift = np.linalg.norm(c_post - c_pre)
        print(f"  {cls}: centroid shift = {shift:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", default=os.path.expanduser("~/medclip_embeddings"))
    parser.add_argument("--out_dir", default=os.path.expanduser("~/medclip_embeddings/plots"))
    parser.add_argument("--k", type=int, default=10, help="k for neighborhood overlap (S_k)")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pre = load_split(args.emb_dir, "z_pre")
    if pre is None:
        raise FileNotFoundError(
            f"No z_pre_embeddings.npy found in {args.emb_dir}. "
            "Run extract_medclip_embeddings.py first."
        )
    post = load_split(args.emb_dir, "z_post")

    print("=" * 60)
    print("STEP 1: normalization check")
    check_normalization(pre["embeddings"], "z_pre")
    if post:
        check_normalization(post["embeddings"], "z_post")

    print("\n" + "=" * 60)
    print("STEP 2: PCA variance")
    plot_pca_variance(pre["embeddings"], os.path.join(args.out_dir, "pca_variance_pre.png"), "z_pre")
    if post:
        plot_pca_variance(post["embeddings"], os.path.join(args.out_dir, "pca_variance_post.png"), "z_post")

    print("\n" + "=" * 60)
    print("STEP 3: class separability (cosine similarity)")
    class_separability(pre["embeddings"], pre["labels"], "z_pre")
    if post:
        class_separability(post["embeddings"], post["labels"], "z_post")

    print("\n" + "=" * 60)
    print("STEP 4: 2D projection")
    coords_pre = project_2d(pre["embeddings"])
    plot_single_projection(coords_pre, pre["labels"],
                            os.path.join(args.out_dir, "projection_pre.png"), "z_pre")
    print(f"Saved: {args.out_dir}/projection_pre.png")

    if post is None:
        print("\nNo z_post found yet — showing pre-fine-tuning baseline only.")
        print("Once you fine-tune and re-extract embeddings as z_post_*.npy, "
              "re-run this script to get drift/trajectory/S_k analysis.")
        print(f"\nAll plots saved to: {args.out_dir}")
        return

    print("\n" + "=" * 60)
    print("STEP 5: drift metrics (eq. 5 / eq. 6 style)")
    l2_drift, cos_sim = compute_drift_metrics(pre, post)

    print("\n" + "=" * 60)
    print("STEP 6: neighborhood overlap (S_k, eq. 11 style)")
    overlaps = knn_overlap(pre, post, k=args.k)

    print("\n" + "=" * 60)
    print("STEP 7: class centroid shift")
    plot_centroid_shift(pre, post, args.out_dir)

    print("\n" + "=" * 60)
    print("STEP 8: joint UMAP + Procrustes trajectory plot")
    plot_joint_drift(pre, post, l2_drift, os.path.join(args.out_dir, "drift_trajectories.png"))
    print(f"Saved: {args.out_dir}/drift_trajectories.png")

    # Save per-sample metrics to CSV for later use (e.g. correlating with attack success)
    import csv
    csv_path = os.path.join(args.out_dir, "per_sample_drift.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "l2_drift", "cosine_sim", "knn_overlap"])
        for p, lbl, d, c, o in zip(pre["paths"], pre["labels"], l2_drift, cos_sim, overlaps):
            writer.writerow([p, lbl, d, c, o])
    print(f"Saved per-sample metrics: {csv_path}")

    print(f"\nAll plots saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
