#!/usr/bin/env python3
"""
sweep_finetune.py

Runs finetune_medclip.py once per (lambda1, lambda2, lambda3) combination in
a grid, calling it as a subprocess each time. Built for the "run the full
lambda sweep (~100 configs) once on exp_4" step in the project's design.

Resumable by construction: before each combo, it checks runs_log.csv for a
row already matching (exp_name, lambda1, lambda2, lambda3, seed) and skips
it. So if the sweep gets interrupted (crash, closed terminal, a laptop that
didn't actually sleep the way you expected), rerunning the exact same
sweep_finetune.py command picks up where it left off instead of redoing
finished configs.

A single bad config does NOT abort the whole sweep -- if one run fails, it's
logged to sweep_failures.log and the sweep moves on to the next combo.

Usage (defaults to a 5x5x5=125-combo grid, edit --lambda-values to change it):
    python sweep_finetune.py \
        --split-json ../data/experiments/exp_4/split.json \
        --exp-name exp_4 \
        --base-checkpoint ../checkpoints/base/model_weights.pt \
        --repo-root .. \
        --epochs 5 --batch-size 16 --lr 1e-5 --seed 42

Dry run first to see the plan without training anything:
    python sweep_finetune.py --split-json ... --exp-name exp_4 --dry-run
"""

import argparse
import csv
import itertools
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-json", required=True)
    p.add_argument("--exp-name", required=True)
    p.add_argument("--base-checkpoint", default="../checkpoints/base/model_weights.pt")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--distance", default="euclidean", choices=["euclidean", "cosine"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--checkpoints-root", default="../checkpoints")
    p.add_argument("--runs-log", default="../runs_log.csv")
    p.add_argument(
        "--lambda-values", default="0.0,0.1,0.5,1.0,2.0",
        help="Comma-separated values applied to all three lambdas as the grid axis "
             "(default gives a 5x5x5=125-combo grid). Edit this to change the grid.",
    )
    p.add_argument(
        "--finetune-script", default=str(Path(__file__).with_name("finetune_medclip.py")),
        help="Path to finetune_medclip.py (defaults to a file of that name next to this script)",
    )
    p.add_argument("--failures-log", default="sweep_failures.log")
    p.add_argument("--dry-run", action="store_true", help="Print the planned combos and exit, run nothing")
    return p.parse_args()


def load_completed_combos(runs_log_path: str, exp_name: str, seed: int):
    """Returns a set of (round(lambda1,6), round(lambda2,6), round(lambda3,6))
    tuples already present in runs_log.csv for this exp_name/seed -- these
    are skipped. Rounding avoids float-formatting mismatches (e.g. '1.0' vs
    '1.000000') between how this sweep and a manual run wrote the CSV."""
    completed = set()
    if not os.path.exists(runs_log_path):
        return completed
    with open(runs_log_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if row["exp_name"] != exp_name or int(row["seed"]) != seed:
                    continue
                key = (
                    round(float(row["lambda1"]), 6),
                    round(float(row["lambda2"]), 6),
                    round(float(row["lambda3"]), 6),
                )
                completed.add(key)
            except (KeyError, ValueError):
                continue  # malformed row -- don't let it crash the sweep
    return completed


def log_failure(failures_log: str, lam1, lam2, lam3, returncode):
    with open(failures_log, "a") as f:
        f.write(
            f"{datetime.now(timezone.utc).isoformat()} "
            f"lambda1={lam1} lambda2={lam2} lambda3={lam3} "
            f"exit_code={returncode}\n"
        )


def main():
    args = parse_args()

    if not os.path.exists(args.finetune_script):
        raise FileNotFoundError(
            f"finetune_medclip.py not found at {args.finetune_script} -- "
            f"pass --finetune-script if it's not next to this file."
        )

    lambda_values = [float(v) for v in args.lambda_values.split(",")]
    all_combos = list(itertools.product(lambda_values, repeat=3))
    completed = load_completed_combos(args.runs_log, args.exp_name, args.seed)

    remaining = [c for c in all_combos if tuple(round(v, 6) for v in c) not in completed]

    print(f"[sweep] grid: {len(lambda_values)}^3 = {len(all_combos)} total configs")
    print(f"[sweep] already completed (per {args.runs_log}): {len(all_combos) - len(remaining)}")
    print(f"[sweep] remaining to run: {len(remaining)}")

    if args.dry_run:
        for l1, l2, l3 in remaining:
            print(f"  lambda1={l1} lambda2={l2} lambda3={l3}")
        return

    n_failed = 0
    for i, (lam1, lam2, lam3) in enumerate(remaining, start=1):
        print(f"\n[sweep] [{i}/{len(remaining)}] lambda1={lam1} lambda2={lam2} lambda3={lam3}")
        cmd = [
            sys.executable, args.finetune_script,
            "--split-json", args.split_json,
            "--exp-name", args.exp_name,
            "--base-checkpoint", args.base_checkpoint,
            "--repo-root", args.repo_root,
            "--lambda1", str(lam1), "--lambda2", str(lam2), "--lambda3", str(lam3),
            "--distance", args.distance,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--seed", str(args.seed),
            "--num-workers", str(args.num_workers),
            "--checkpoints-root", args.checkpoints_root,
            "--runs-log", args.runs_log,
        ]
        if args.device:
            cmd += ["--device", args.device]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[sweep] config lambda1={lam1} lambda2={lam2} lambda3={lam3} FAILED "
                  f"(exit {result.returncode}) -- logged, continuing to next config")
            log_failure(args.failures_log, lam1, lam2, lam3, result.returncode)
            n_failed += 1

    n_run = len(remaining) - n_failed
    print(f"\n[sweep] done -- {n_run} succeeded, {n_failed} failed "
          f"(see {args.failures_log}), {len(all_combos) - len(remaining)} were already complete")
    if n_failed:
        print(f"[sweep] rerun the exact same command to retry failed configs -- "
              f"they aren't in runs_log.csv, so they won't be skipped")


if __name__ == "__main__":
    main()
