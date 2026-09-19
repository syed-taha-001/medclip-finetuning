#!/usr/bin/env python
"""
scripts/extract_embeddings.py

Implements Eqs. 2-3 of the proposal: z = f_theta(x).

Takes a checkpoint (base or a specific fine-tuning run) plus a split.json's
image manifest, runs a forward pass through MedCLIP's image encoder, and
saves the resulting embedding matrix Z in R^{B x d} plus a _meta.json
sidecar recording exactly which checkpoint/split/subset produced it.

Usage
-----
# z_pre: base model against an experiment's held-out set
python scripts/extract_embeddings.py \
    --checkpoint checkpoints/base/model_weights.pt \
    --split data/experiments/exp_4/split.json \
    --subset d1 \
    --run-id base \
    --exp-name exp_4

# z_post: a specific fine-tuning run, both subsets
python scripts/extract_embeddings.py \
    --checkpoint checkpoints/exp_4/run_003/model_weights.pt \
    --split data/experiments/exp_4/split.json \
    --subset both \
    --run-id run_003 \
    --exp-name exp_4

Output layout
-------------
embeddings/<exp-name>/<run-id>/
    z_<subset>.npy        # float32 array, shape (N, d)
    labels_<subset>.npy   # int64 array, shape (N,) — 0=NORMAL, 1=PNEUMONIA
    paths_<subset>.json   # list[str], repo-root-relative, same order as rows
    meta.json             # provenance: checkpoint, split, subset(s), model
                           # info, preprocessing description, timestamp

Design constraints this script honors (see implementation-state.md)
---------------------------------------------------------------------
- Preprocessing mirrors finetune_medclip.py exactly:
  PIL.Image -> .convert("L") -> MedCLIPProcessor(images=..., return_tensors="pt")
- Model is always constructed as MedCLIPModel(vision_cls=MedCLIPVisionModelViT).
  NOTE: despite the class name, MedCLIPVisionModelViT is the verified-correct
  class for this checkpoint (it is actually a Swin backbone internally --
  MedCLIPVisionModel, the "other" class, is actually ResNet50). Do not swap
  this without re-verifying key match counts against the checkpoint.
- Checkpoints are raw state_dict() files (no optimizer state) -> loaded via
  plain model.load_state_dict(...).
- Labels are inferred from each image path's parent directory name
  (NORMAL / PNEUMONIA), not from split.json, since the exact split.json
  schema written by create_splits.py hasn't been pasted into this session --
  this keeps the script correct regardless of whether split.json stores
  bare path strings or {"path": ..., "label": ...} dicts.
- Output is overwrite-protected by default, matching create_splits.py.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from medclip import MedCLIPModel, MedCLIPProcessor, MedCLIPVisionModelViT
except ImportError as e:
    print(
        "ERROR: could not import medclip. Activate the 'medclip' conda env "
        f"before running this script.\n  {e}",
        file=sys.stderr,
    )
    sys.exit(1)


LABEL_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}

# Candidate key names for the two halves of a split.json, in priority order.
PRIME_KEYS = ["d1_prime", "d1'", "train", "finetune", "d1_train"]
HOLDOUT_KEYS = ["d1", "holdout", "eval", "test", "d1_eval", "d1_holdout"]


# --------------------------------------------------------------------------- #
# split.json parsing
# --------------------------------------------------------------------------- #
def _entries_to_paths(entries):
    """Normalize a list of split.json entries (bare strings or dicts with a
    'path'/'image'/'file' key) into a flat list of path strings."""
    paths = []
    for e in entries:
        if isinstance(e, str):
            paths.append(e)
        elif isinstance(e, dict):
            for key in ("path", "image", "file", "filepath", "img_path"):
                if key in e:
                    paths.append(e[key])
                    break
            else:
                raise ValueError(
                    f"split.json entry is a dict with no recognizable path "
                    f"key (looked for path/image/file/filepath/img_path): {e}"
                )
        else:
            raise ValueError(f"Unrecognized split.json entry type: {type(e)} -> {e}")
    return paths


def load_split(split_path: Path, subset: str):
    """Returns dict subset_name -> list[str] (repo-root-relative paths).
    subset is one of 'd1_prime', 'd1', 'both'."""
    with open(split_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected split.json to be a JSON object, got {type(data)}. "
            f"If create_splits.py's schema differs from what this script "
            f"expects, update PRIME_KEYS/HOLDOUT_KEYS at the top of this file."
        )

    def find(keys):
        for k in keys:
            if k in data:
                return _entries_to_paths(data[k])
        raise KeyError(
            f"None of the expected keys {keys} found in {split_path}. "
            f"Actual top-level keys: {list(data.keys())}. Update "
            f"PRIME_KEYS/HOLDOUT_KEYS in extract_embeddings.py to match "
            f"create_splits.py's real schema."
        )

    out = {}
    if subset in ("d1_prime", "both"):
        out["d1_prime"] = find(PRIME_KEYS)
    if subset in ("d1", "both"):
        out["d1"] = find(HOLDOUT_KEYS)
    return out


def infer_label(path_str: str) -> int:
    """Label comes from the immediate parent directory name
    (data/raw/kermany/{NORMAL,PNEUMONIA}/xxx.jpeg), case-insensitive."""
    parent = Path(path_str).parent.name.upper()
    if parent not in LABEL_TO_IDX:
        raise ValueError(
            f"Could not infer NORMAL/PNEUMONIA label from path '{path_str}' "
            f"(parent dir '{parent}' is not NORMAL or PNEUMONIA)."
        )
    return LABEL_TO_IDX[parent]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ImagePathDataset(Dataset):
    """Loads one image at a time, applies .convert("L"), runs it through the
    MedCLIP processor individually, and returns a squeezed pixel_values
    tensor. Processing one-at-a-time (instead of batched processor calls)
    sidesteps any ambiguity in whether MedCLIPProcessor batches PIL lists
    the way HF processors do."""

    def __init__(self, repo_root: Path, rel_paths, processor):
        self.repo_root = repo_root
        self.rel_paths = rel_paths
        self.processor = processor

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, idx):
        rel = self.rel_paths[idx]
        abs_path = self.repo_root / rel
        img = Image.open(abs_path).convert("L")
        processed = self.processor(images=img, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)  # (C, H, W)
        label = infer_label(rel)
        return pixel_values, label, rel


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_model(checkpoint_path: Path, device: str):
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # strict=True raises on any missing/unexpected key, matching the
    # 0-missing/0-unexpected verification already done on this checkpoint.
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    return model


def get_image_embeddings(model, pixel_values):
    """z = f_theta(x). Tries the documented MedCLIP API first
    (model.encode_image), falls back to the raw vision tower + projection
    if encode_image isn't present on this package version."""
    if hasattr(model, "encode_image"):
        return model.encode_image(pixel_values)
    if hasattr(model, "vision_model") and hasattr(model, "visual_projection"):
        feats = model.vision_model(pixel_values)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        return model.visual_projection(feats)
    raise AttributeError(
        "MedCLIPModel has neither .encode_image() nor "
        ".vision_model + .visual_projection. Inspect the installed medclip "
        "package (dir(model)) and update get_image_embeddings() accordingly."
    )


# --------------------------------------------------------------------------- #
# Main extraction routine
# --------------------------------------------------------------------------- #
def extract(model, rel_paths, processor, repo_root, device, batch_size, num_workers):
    ds = ImagePathDataset(repo_root, rel_paths, processor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    n_batches = len(dl)
    all_z, all_labels, all_paths = [], [], []
    start = time.time()
    with torch.no_grad():
        for i, (pixel_values, labels, paths) in enumerate(
            tqdm(dl, total=n_batches, desc="extracting", unit="batch")
        ):
            pixel_values = pixel_values.to(device)
            z = get_image_embeddings(model, pixel_values)
            all_z.append(z.cpu().numpy().astype(np.float32))
            all_labels.append(labels.numpy().astype(np.int64))
            all_paths.extend(paths)

            if device == "cpu" and i == 4:
                # After a few batches, print a rough ETA so a slow CPU run
                # doesn't look identical to a genuine hang.
                elapsed = time.time() - start
                per_batch = elapsed / (i + 1)
                eta_min = (per_batch * (n_batches - i - 1)) / 60
                tqdm.write(
                    f"[extract_embeddings] CPU device — rough ETA for this "
                    f"subset: ~{eta_min:.1f} min remaining"
                )

    Z = np.concatenate(all_z, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    return Z, labels_arr, all_paths


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path, help="Path to model_weights.pt (raw state_dict)")
    p.add_argument("--split", required=True, type=Path, help="Path to data/experiments/exp_X/split.json")
    p.add_argument("--subset", required=True, choices=["d1_prime", "d1", "both"])
    p.add_argument("--exp-name", required=True, help="e.g. exp_4 (used in output path)")
    p.add_argument("--run-id", required=True, help="e.g. 'base' or 'run_003' (used in output path)")
    p.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Root that split.json paths are relative to")
    p.add_argument("--output-root", type=Path, default=Path("embeddings"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing output dir")
    args = p.parse_args()

    out_dir = args.output_root / args.exp_name / args.run_id
    if out_dir.exists() and not args.overwrite:
        existing = list(out_dir.glob("z_*.npy"))
        if existing:
            print(
                f"ERROR: {out_dir} already has output ({[f.name for f in existing]}). "
                f"Pass --overwrite to replace it.",
                file=sys.stderr,
            )
            sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract_embeddings] loading checkpoint: {args.checkpoint}")
    processor = MedCLIPProcessor()
    model = load_model(args.checkpoint, args.device)
    print(f"[extract_embeddings] model on {args.device}, eval mode, grads frozen")

    subsets = load_split(args.split, args.subset)

    meta = {
        "checkpoint": str(args.checkpoint),
        "split": str(args.split),
        "subset_requested": args.subset,
        "exp_name": args.exp_name,
        "run_id": args.run_id,
        "device": args.device,
        "batch_size": args.batch_size,
        "preprocessing": "PIL.Image.convert('L') -> MedCLIPProcessor(images=..., return_tensors='pt')",
        "model": "MedCLIPModel(vision_cls=MedCLIPVisionModelViT)",
        "extracted_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "subsets_written": {},
    }

    for subset_name, rel_paths in subsets.items():
        print(f"[extract_embeddings] extracting subset='{subset_name}' ({len(rel_paths)} images)")
        Z, labels_arr, paths_used = extract(
            model, rel_paths, processor, args.repo_root, args.device,
            args.batch_size, args.num_workers,
        )

        np.save(out_dir / f"z_{subset_name}.npy", Z)
        np.save(out_dir / f"labels_{subset_name}.npy", labels_arr)
        with open(out_dir / f"paths_{subset_name}.json", "w") as f:
            json.dump(paths_used, f, indent=2)

        n_normal = int((labels_arr == 0).sum())
        n_pneumonia = int((labels_arr == 1).sum())
        print(
            f"[extract_embeddings]   -> Z shape {Z.shape}, "
            f"NORMAL={n_normal}, PNEUMONIA={n_pneumonia}, "
            f"saved to {out_dir}/z_{subset_name}.npy"
        )
        meta["subsets_written"][subset_name] = {
            "n_samples": int(Z.shape[0]),
            "embedding_dim": int(Z.shape[1]),
            "n_normal": n_normal,
            "n_pneumonia": n_pneumonia,
        }

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[extract_embeddings] done. meta.json + arrays written to {out_dir}")


if __name__ == "__main__":
    main()
