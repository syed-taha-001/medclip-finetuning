#!/usr/bin/env python3
"""
compare_embeddings.py

Post-hoc comparison of a pre-adaptation embedding set (z_pre, base model)
against a post-adaptation embedding set (z_post, one fine-tuned run) on the
held-out `d1` evaluation subset. Uses the confirmed metrics.py functions:

    - per_sample_drift(z_pre, z_post)          Eq. 10 adapted: pre- vs.
                                                post-adaptation on the same
                                                clean sample (not clean vs.
                                                attacked)
    - neighborhood_stability(z_pre, z_post, k) Eq. 11 adapted: S_k(x_i) =
                                                |N_k(x_i) intersect N'_k(x_i)| / k

This script does NOT train or re-run the model, and does NOT touch
checkpoints. It only compares two already-extracted .npy embedding arrays
(produced by extract_embeddings.py). "Run" = one finetune_medclip.py
session; this script is a separate, later step per project convention.

Two outputs are written, deliberately kept separate (plots are for human
inspection, JSON metric files are for formula application/aggregation):

    1. <output-dir>/per_sample_drift.csv    -- per-sample drift + S_k, for
                                                plotting
    2. <output-dir>/comparison_metrics.json -- summary numbers (overall +
                                                per-class), for cross-run
                                                aggregation / the lambda
                                                sweep table

Optionally patches a matching row in runs_log.csv with the summary numbers
(--update-runs-log), so the sweep table ends up with drift/S_k columns
alongside the accuracy/F1/AUC that finetune_medclip.py already logs.

--------------------------------------------------------------------------
NOT YET VERIFIED against the real files on disk (flagging per the project's
"distinguish confirmed-executed from confident-but-unrun" standard):

  1. Row order: this assumes embeddings/<...>/z_<subset>.npy rows are in the
     SAME ORDER as split.json[subset]. extract_embeddings.py doesn't appear
     to reorder anything, but this has not been spot-checked. The script
     hard-fails on a *count* mismatch; it cannot catch a same-count
     reordering. Worth a manual spot-check (e.g. confirm row 0's inferred
     label matches the first path in split.json[subset]) the first time
     this runs for real.
  2. runs_log.csv's run-identifying column name is assumed to be
     "checkpoint_path" (--match-column default). Not confirmed against the
     actual CSV header written by finetune_medclip.py -- the script errors
     out cleanly (lists real columns) rather than guessing if it's wrong.
  3. split.json[subset] entry shape: assumed to be a list of either bare
     path strings, or dicts with a "path"/"image_path"/"img_path" key.
--------------------------------------------------------------------------

Usage:
    python compare_embeddings.py \\
        --z-pre embeddings/exp_4/base/z_d1.npy \\
        --z-post embeddings/exp_4/run_001/z_d1.npy \\
        --split-json data/splits/exp_4/split.json \\
        --repo-root .. \\
        --exp-name exp_4 --run-name run_001 \\
        --k 5 10 20 50 \\
        --update-runs-log --runs-log runs_log.csv \\
        --match-value checkpoints/exp_4/run_001/model_weights.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

# metrics.py is expected alongside this script (same convention as
# finetune_medclip.py / test_gradient_flow.py importing it directly).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import per_sample_drift, neighborhood_stability  # noqa: E402


LABELS = ("NORMAL", "PNEUMONIA")


def infer_label(path_str: str) -> str:
    """Same rule extract_embeddings.py uses: the image's parent directory name."""
    parent = Path(path_str).parent.name.upper()
    if parent not in LABELS:
        raise ValueError(
            f"Cannot infer NORMAL/PNEUMONIA label from path: {path_str} "
            f"(parent dir '{parent}' not in {LABELS})"
        )
    return parent


def load_subset_paths(split_json_path: Path, subset: str) -> list:
    with open(split_json_path) as f:
        split = json.load(f)
    if subset not in split:
        raise KeyError(
            f"'{subset}' not found in {split_json_path} "
            f"(top-level keys: {list(split.keys())})"
        )
    entries = split[subset]
    paths = []
    for e in entries:
        if isinstance(e, str):
            paths.append(e)
        elif isinstance(e, dict):
            for key in ("path", "image_path", "img_path"):
                if key in e:
                    paths.append(e[key])
                    break
            else:
                raise KeyError(f"No path-like key found in split entry: {e}")
        else:
            raise TypeError(f"Unexpected split entry type: {type(e)}")
    return paths


def load_labels(split_json_path: Path, subset: str, repo_root: Path) -> np.ndarray:
    paths = load_subset_paths(split_json_path, subset)
    labels = [infer_label(str(repo_root / p)) for p in paths]
    return np.array(labels)


def per_class_summary(values: np.ndarray, labels: np.ndarray) -> dict:
    out = {
        "overall_mean": float(np.mean(values)),
        "overall_std": float(np.std(values)),
    }
    for label in LABELS:
        mask = labels == label
        if mask.sum() == 0:
            continue
        out[f"{label.lower()}_mean"] = float(np.mean(values[mask]))
        out[f"{label.lower()}_std"] = float(np.std(values[mask]))
        out[f"{label.lower()}_n"] = int(mask.sum())
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--z-pre", required=True, type=Path,
                    help="z_pre .npy (base model), e.g. embeddings/exp_4/base/z_d1.npy")
    p.add_argument("--z-post", required=True, type=Path,
                    help="z_post .npy (fine-tuned run), e.g. embeddings/exp_4/run_001/z_d1.npy")
    p.add_argument("--split-json", required=True, type=Path,
                    help="split.json for this experiment (for label inference)")
    p.add_argument("--subset", default="d1",
                    help="split.json key these embeddings correspond to (default: d1, the held-out eval set)")
    p.add_argument("--repo-root", default=Path("."), type=Path,
                    help="Root that split.json's relative paths are joined against "
                         "(use '..' when running from scripts/, same as finetune_medclip.py)")
    p.add_argument("--k", type=int, nargs="+", default=[5, 10, 20, 50],
                    help="k value(s) for neighborhood_stability / S_k")
    p.add_argument("--exp-name", required=True, help="e.g. exp_4")
    p.add_argument("--run-name", required=True,
                    help="e.g. run_001 -- used for output paths and JSON metadata only")
    p.add_argument("--output-dir", type=Path, default=None,
                    help="Where to write per_sample_drift.csv / comparison_metrics.json "
                         "(default: same directory as --z-post)")
    p.add_argument("--update-runs-log", action="store_true",
                    help="Patch a matching row in --runs-log with the summary numbers "
                         "(requires pandas; verify --match-column against the real CSV header first)")
    p.add_argument("--runs-log", type=Path, default=Path("runs_log.csv"))
    p.add_argument("--match-column", default="checkpoint_path",
                    help="Column in runs_log.csv used to find the row for this run")
    p.add_argument("--match-value", default=None,
                    help="Value to match in --match-column (e.g. the checkpoint path z_post "
                         "came from); required with --update-runs-log")
    args = p.parse_args()

    if args.update_runs_log and args.match_value is None:
        p.error("--update-runs-log requires --match-value")
    if args.update_runs_log and pd is None:
        p.error("--update-runs-log requires pandas, which isn't importable here")

    z_pre = np.load(args.z_pre)
    z_post = np.load(args.z_post)
    if z_pre.shape != z_post.shape:
        p.error(f"Shape mismatch: z_pre {z_pre.shape} vs z_post {z_post.shape}")

    labels = load_labels(args.split_json, args.subset, args.repo_root)
    if len(labels) != z_pre.shape[0]:
        p.error(
            f"Label count ({len(labels)}) from split.json['{args.subset}'] does not match "
            f"embedding count ({z_pre.shape[0]}) -- refusing to guess at the correspondence."
        )

    n_normal = int((labels == "NORMAL").sum())
    n_pneu = int((labels == "PNEUMONIA").sum())
    print(f"[{args.exp_name}/{args.run_name}] {z_pre.shape[0]} samples "
          f"({n_normal} NORMAL / {n_pneu} PNEUMONIA)")

    # --- Eq. 10 adapted: per-sample drift ---
    drift = per_sample_drift(z_pre, z_post)
    drift_summary = per_class_summary(drift, labels)
    print(f"  drift: mean={drift_summary['overall_mean']:.4f} std={drift_summary['overall_std']:.4f}")

    # --- Eq. 11 adapted: neighborhood stability S_k, for each requested k ---
    sk_arrays = {}
    sk_summaries = {}
    for k in args.k:
        if k >= z_pre.shape[0]:
            print(f"  skipping k={k}: >= sample count ({z_pre.shape[0]})", file=sys.stderr)
            continue
        try:
            sk = neighborhood_stability(z_pre, z_post, k)
        except ValueError as e:
            print(f"  k={k} raised ValueError from metrics.py: {e}", file=sys.stderr)
            continue
        sk_arrays[k] = sk
        sk_summaries[k] = per_class_summary(sk, labels)
        print(f"  S_{k}: mean={sk_summaries[k]['overall_mean']:.4f} std={sk_summaries[k]['overall_std']:.4f}")

    out_dir = args.output_dir or args.z_post.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- per-sample CSV, for plotting ---
    per_sample_path = out_dir / "per_sample_drift.csv"
    with open(per_sample_path, "w") as f:
        header = ["index", "label", "drift"] + [f"s_{k}" for k in sk_arrays]
        f.write(",".join(header) + "\n")
        for i in range(z_pre.shape[0]):
            row = [str(i), labels[i], f"{drift[i]:.6f}"]
            row += [f"{sk_arrays[k][i]:.6f}" for k in sk_arrays]
            f.write(",".join(row) + "\n")
    print(f"  wrote {per_sample_path}")

    # --- summary JSON, for aggregation across the lambda sweep ---
    summary = {
        "exp_name": args.exp_name,
        "run_name": args.run_name,
        "z_pre_path": str(args.z_pre),
        "z_post_path": str(args.z_post),
        "n_samples": int(z_pre.shape[0]),
        "drift": drift_summary,
        "neighborhood_stability": {str(k): v for k, v in sk_summaries.items()},
    }
    metrics_path = out_dir / "comparison_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {metrics_path}")

    # --- optionally patch runs_log.csv ---
    if args.update_runs_log:
        if not args.runs_log.exists():
            p.error(f"--runs-log not found: {args.runs_log}")
        df = pd.read_csv(args.runs_log)
        if args.match_column not in df.columns:
            p.error(
                f"'{args.match_column}' not in {args.runs_log} columns "
                f"({list(df.columns)}) -- pass the correct --match-column"
            )
        mask = df[args.match_column] == args.match_value
        if mask.sum() == 0:
            p.error(f"No row in {args.runs_log} where {args.match_column} == {args.match_value!r}")
        if mask.sum() > 1:
            p.error(f"{mask.sum()} rows matched {args.match_column} == {args.match_value!r}, expected exactly 1")

        df.loc[mask, "drift_mean"] = drift_summary["overall_mean"]
        df.loc[mask, "drift_normal_mean"] = drift_summary.get("normal_mean")
        df.loc[mask, "drift_pneumonia_mean"] = drift_summary.get("pneumonia_mean")
        for k, s in sk_summaries.items():
            df.loc[mask, f"s_{k}_mean"] = s["overall_mean"]

        df.to_csv(args.runs_log, index=False)
        print(f"  patched {args.runs_log} (matched on {args.match_column} == {args.match_value!r})")


if __name__ == "__main__":
    main()
