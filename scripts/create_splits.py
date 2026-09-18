"""
create_splits.py

Builds the experiment splits for the Kermany chest X-ray fine-tuning study.

Pipeline:
  Layer 1 - Partition the full Kermany pool (NORMAL + PNEUMONIA) into 3
            disjoint, class-stratified thirds using StratifiedKFold, so
            each partition preserves the overall NORMAL:PNEUMONIA ratio.
  Layer 2 - Within each partition, stratified split into:
                d1_prime (fine-tuning train set)
                d1       (held-out evaluation set)
            using --holdout-frac as the fraction assigned to d1
            (default 0.3 -> 70/30 train/holdout).
  exp_4   - NOT independently re-split. Built as the union of
            exp_1/2/3's d1_prime sets and d1 sets respectively, so the
            "does the global-search winner generalize" comparison in
            compare_experiments.py is not confounded by a differently
            drawn held-out set.

Images are never copied or moved. This script only records file paths
(relative to the repository root) into data/experiments/<exp>/split.json.
Run this script from the repository root so the stored paths resolve
correctly on every machine (paths are NOT stored as absolute paths,
since usernames/home directories differ between the college PC and
personal laptop).
"""

import argparse
import json
import sys
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
CLASS_NAMES = ("NORMAL", "PNEUMONIA")


def load_class_paths(data_root: Path, class_name: str) -> list:
    """Return a sorted, deterministic list of image paths for one class.

    Sorting BEFORE any shuffling is what makes the later seeded shuffle
    reproducible across machines / filesystems (directory iteration
    order is not guaranteed to be stable otherwise). Non-image files
    (.gitkeep, thumbnails, stray text files) are excluded by extension.
    """
    class_dir = data_root / class_name
    if not class_dir.is_dir():
        raise FileNotFoundError(
            f"Expected class directory not found: {class_dir}. "
            f"Did you run setup_structure.sh and copy the dataset in?"
        )
    paths = sorted(
        p.as_posix()
        for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(
            f"No image files found in {class_dir}. "
            f"Expected extensions: {sorted(IMAGE_EXTENSIONS)}."
        )
    return paths


def build_partitions(all_paths: list, all_labels: list, seed: int):
    """Layer 1: split the full pool into 3 disjoint, class-stratified
    partitions via StratifiedKFold. Each partition's held-out ("test")
    fold across the 3 splits is pairwise disjoint, and their union is
    the full dataset -- exactly the non-overlap property needed for
    exp_1 / exp_2 / exp_3.
    """
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    partitions = []
    for _, part_idx in skf.split(all_paths, all_labels):
        part_paths = [all_paths[i] for i in part_idx]
        part_labels = [all_labels[i] for i in part_idx]
        partitions.append((part_paths, part_labels))
    return partitions


def stratified_holdout(paths: list, labels: list, holdout_frac: float, seed: int):
    """Layer 2: stratified split of one partition into d1_prime / d1.

    sklearn's train_test_split(X, y, test_size=f) returns
    (X_train, X_test, y_train, y_test), where the "train" side gets
    fraction (1 - f) and the "test" side gets fraction f. Train maps to
    d1_prime (larger, used for fine-tuning); test maps to d1 (smaller,
    held out for evaluation).
    """
    d1_prime_paths, d1_paths, d1_prime_labels, d1_labels = train_test_split(
        paths,
        labels,
        test_size=holdout_frac,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )
    return d1_prime_paths, d1_prime_labels, d1_paths, d1_labels


def label_balance(labels: list) -> dict:
    return {
        "NORMAL": labels.count("NORMAL"),
        "PNEUMONIA": labels.count("PNEUMONIA"),
        "total": len(labels),
    }


def check_overwrite_safety(output_dir: Path, exp_names: list, overwrite: bool):
    """Check ALL four target files before writing ANY of them, so a
    conflict on (say) exp_1 can't leave exp_2/3/4 written while exp_1
    is left stale -- either every split.json gets regenerated together,
    or none of them do.
    """
    if overwrite:
        return
    existing = [
        str(output_dir / exp / "split.json")
        for exp in exp_names
        if (output_dir / exp / "split.json").exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing split.json file(s):\n  "
            + "\n  ".join(existing)
            + "\nRegenerating a split silently invalidates any embeddings or "
              "checkpoints already computed against the old one. Pass "
              "--overwrite if you intend to regenerate all four experiment splits."
        )


def write_split_json(output_dir: Path, exp_name: str,
                      d1_prime_paths: list, d1_prime_labels: list,
                      d1_paths: list, d1_labels: list,
                      seed: int, holdout_frac: float) -> dict:
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    split_data = {
        "d1_prime": sorted(d1_prime_paths),
        "d1": sorted(d1_paths),
        "meta": {
            "seed": seed,
            "holdout_frac": holdout_frac,
            "path_convention": "relative to repository root; run downstream scripts from repo root",
            "d1_prime_balance": label_balance(d1_prime_labels),
            "d1_balance": label_balance(d1_labels),
        },
    }
    with open(exp_dir / "split.json", "w") as f:
        json.dump(split_data, f, indent=2)
    return split_data


def run_self_checks(narrow_splits: dict, exp4_d1_prime: list, exp4_d1: list, all_paths: list):
    """Fail loudly on any inconsistency rather than silently writing a
    corrupted set of splits. Verifies:
      1. exp_1/2/3 are pairwise disjoint and their union covers the
         full source pool exactly once (no image lost, none duplicated).
      2. d1_prime and d1 don't overlap within any single experiment.
      3. exp_4's d1_prime / d1 are exact, duplicate-free unions of
         exp_1/2/3's respective sets.
    """
    errors = []

    seen = set()
    for exp_name in ("exp_1", "exp_2", "exp_3"):
        d1p = narrow_splits[exp_name]["d1_prime_paths"]
        d1 = narrow_splits[exp_name]["d1_paths"]
        exp_all = set(d1p) | set(d1)
        if len(exp_all) != len(d1p) + len(d1):
            errors.append(f"{exp_name}: d1_prime and d1 overlap internally.")
        overlap = seen & exp_all
        if overlap:
            errors.append(f"{exp_name}: overlaps with a previous partition ({len(overlap)} shared paths).")
        seen |= exp_all

    if seen != set(all_paths):
        missing = set(all_paths) - seen
        extra = seen - set(all_paths)
        if missing:
            errors.append(f"{len(missing)} images from the full pool are missing across exp_1/2/3.")
        if extra:
            errors.append(f"{len(extra)} unexpected paths appear in exp_1/2/3 that aren't in the source pool.")

    expected_exp4_d1_prime = (
        set(narrow_splits["exp_1"]["d1_prime_paths"])
        | set(narrow_splits["exp_2"]["d1_prime_paths"])
        | set(narrow_splits["exp_3"]["d1_prime_paths"])
    )
    expected_exp4_d1 = (
        set(narrow_splits["exp_1"]["d1_paths"])
        | set(narrow_splits["exp_2"]["d1_paths"])
        | set(narrow_splits["exp_3"]["d1_paths"])
    )
    if set(exp4_d1_prime) != expected_exp4_d1_prime or len(exp4_d1_prime) != len(expected_exp4_d1_prime):
        errors.append("exp_4 d1_prime is not an exact, duplicate-free union of exp_1/2/3's d1_prime sets.")
    if set(exp4_d1) != expected_exp4_d1 or len(exp4_d1) != len(expected_exp4_d1):
        errors.append("exp_4 d1 is not an exact, duplicate-free union of exp_1/2/3's d1 sets.")

    if errors:
        print("\nSELF-CHECK FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        raise AssertionError(
            "create_splits.py self-checks failed; see errors above. "
            "No split.json files were written."
        )

    print("\nSelf-checks passed: exp_1/2/3 are pairwise disjoint and cover the full pool "
          "exactly once; exp_4 is an exact union of exp_1/2/3 with no duplicates.")


def print_summary(exp_name: str, d1_prime_labels: list, d1_labels: list):
    p = label_balance(d1_prime_labels)
    h = label_balance(d1_labels)
    print(f"  {exp_name:<8} d1'={p['total']:>5} (N={p['NORMAL']:>4}, P={p['PNEUMONIA']:>4})   "
          f"d1={h['total']:>5} (N={h['NORMAL']:>4}, P={h['PNEUMONIA']:>4})")


def main():
    parser = argparse.ArgumentParser(
        description="Partition the Kermany dataset into exp_1/2/3 (disjoint thirds) "
                    "and exp_4 (their union), each split into d1_prime / d1."
    )
    parser.add_argument("--data-root", type=str, default="data/raw/kermany",
                         help="Folder containing NORMAL/ and PNEUMONIA/ subfolders. "
                              "Keep this relative to the repo root for cross-machine portability.")
    parser.add_argument("--output-dir", type=str, default="data/experiments",
                         help="Where exp_1..4/split.json files are written.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for partitioning and the holdout split.")
    parser.add_argument("--holdout-frac", type=float, default=0.3,
                         help="Fraction of each experiment held out as d1 (default 0.3 -> 70/30).")
    parser.add_argument("--overwrite", action="store_true",
                         help="Allow overwriting existing split.json files. Off by default: "
                              "regenerating a split silently invalidates any embeddings or "
                              "checkpoints already computed against the old one.")
    args = parser.parse_args()

    if not (0.0 < args.holdout_frac < 1.0):
        parser.error(f"--holdout-frac must be between 0 and 1 (exclusive), got {args.holdout_frac}")

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    exp_names = ["exp_1", "exp_2", "exp_3", "exp_4"]

    if data_root.is_absolute():
        print("WARNING: --data-root is an absolute path. Paths written into split.json will "
              "then also be absolute, which will break on any machine with a different "
              "home directory / username. Prefer a relative path.", file=sys.stderr)

    check_overwrite_safety(output_dir, exp_names, args.overwrite)

    print(f"Loading images from {data_root} ...")
    all_paths, all_labels = [], []
    for class_name in CLASS_NAMES:
        class_paths = load_class_paths(data_root, class_name)
        all_paths.extend(class_paths)
        all_labels.extend([class_name] * len(class_paths))
        print(f"  {class_name}: {len(class_paths)} images")

    total_pool = label_balance(all_labels)
    print(f"Total pool: {total_pool['total']} images "
          f"(NORMAL={total_pool['NORMAL']}, PNEUMONIA={total_pool['PNEUMONIA']})\n")

    print("Building 3 disjoint, class-stratified partitions (Layer 1)...")
    partitions = build_partitions(all_paths, all_labels, seed=args.seed)

    narrow_splits = {}
    print(f"Splitting each partition (holdout_frac={args.holdout_frac}) (Layer 2)...")
    for idx, (part_paths, part_labels) in enumerate(partitions, start=1):
        exp_name = f"exp_{idx}"
        d1p_paths, d1p_labels, d1_paths, d1_labels = stratified_holdout(
            part_paths, part_labels, args.holdout_frac, seed=args.seed
        )
        narrow_splits[exp_name] = {
            "d1_prime_paths": d1p_paths,
            "d1_prime_labels": d1p_labels,
            "d1_paths": d1_paths,
            "d1_labels": d1_labels,
        }

    # exp_4: union of exp_1/2/3, NOT an independent re-split.
    exp4_d1_prime = (
        narrow_splits["exp_1"]["d1_prime_paths"]
        + narrow_splits["exp_2"]["d1_prime_paths"]
        + narrow_splits["exp_3"]["d1_prime_paths"]
    )
    exp4_d1_prime_labels = (
        narrow_splits["exp_1"]["d1_prime_labels"]
        + narrow_splits["exp_2"]["d1_prime_labels"]
        + narrow_splits["exp_3"]["d1_prime_labels"]
    )
    exp4_d1 = (
        narrow_splits["exp_1"]["d1_paths"]
        + narrow_splits["exp_2"]["d1_paths"]
        + narrow_splits["exp_3"]["d1_paths"]
    )
    exp4_d1_labels = (
        narrow_splits["exp_1"]["d1_labels"]
        + narrow_splits["exp_2"]["d1_labels"]
        + narrow_splits["exp_3"]["d1_labels"]
    )

    print("\nRunning self-checks before writing anything to disk...")
    run_self_checks(narrow_splits, exp4_d1_prime, exp4_d1, all_paths)

    all_splits = dict(narrow_splits)
    all_splits["exp_4"] = {
        "d1_prime_paths": exp4_d1_prime,
        "d1_prime_labels": exp4_d1_prime_labels,
        "d1_paths": exp4_d1,
        "d1_labels": exp4_d1_labels,
    }

    print("\nWriting split.json files...")
    for exp_name in exp_names:
        s = all_splits[exp_name]
        write_split_json(
            output_dir, exp_name,
            s["d1_prime_paths"], s["d1_prime_labels"],
            s["d1_paths"], s["d1_labels"],
            args.seed, args.holdout_frac,
        )
        print(f"  wrote {output_dir / exp_name / 'split.json'}")

    print("\nSummary (d1' = fine-tune set, d1 = held-out set, N=NORMAL, P=PNEUMONIA):")
    for exp_name in exp_names:
        s = all_splits[exp_name]
        print_summary(exp_name, s["d1_prime_labels"], s["d1_labels"])

    print("\nAll splits written successfully.")


if __name__ == "__main__":
    main()
