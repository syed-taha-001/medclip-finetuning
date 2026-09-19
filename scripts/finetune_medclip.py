#!/usr/bin/env python3
"""
finetune_medclip.py

Stability-constrained fine-tuning of MedCLIP under embedding-drift
regularization (Eq. 8, shaheer.pdf):

    L_adapt = L_task + lambda1 * L_drift + lambda2 * L_cos + lambda3 * L_nbr

Design decisions this script follows (see project memory: implementation-state.md):

1. z_pre for d1_prime is NEVER loaded from a precomputed .npy file. Instead,
   a frozen copy of the base model is kept loaded alongside the trainable
   one. For every training batch, that exact batch is run through the
   frozen copy (no_grad) to get z_pre, and through the trainable model to
   get z_post -- same batch, same order, zero shuffle/lookup risk. This
   mirrors test_gradient_flow.py, which already proved the pattern works.

2. Preprocessing mirrors extract_embeddings.py / test_gradient_flow.py
   exactly: PIL.Image.open(path).convert("L") -> MedCLIPProcessor(images=...,
   return_tensors="pt")["pixel_values"].squeeze(0). Do not change this
   without changing it everywhere else too.

3. Labels are inferred from each image path's parent directory name
   (NORMAL / PNEUMONIA), never trusted from split.json fields -- same
   convention as extract_embeddings.py.

4. L_nbr pairs are same-class only (NORMAL-NORMAL, PNEUMONIA-PNEUMONIA),
   built from within-batch label pairs via a double loop over batch indices.

5. lambda1 = lambda2 = lambda3 = 0.0 is the naive fine-tuning baseline (no
   geometry regularization) -- an ordinary configuration, not a special case.

6. This script does exactly one job: load base weights, train on d1_prime,
   evaluate once on d1, save a checkpoint, append one row to runs_log.csv.
   It does NOT extract or compare embeddings -- that's extract_embeddings.py's
   and compare_embeddings.py's job, run afterward as separate steps.

7. checkpoints/base/model_weights.pt is read-only, ever -- this script only
   writes to checkpoints/<exp-name>/run_00X/.

Assumes losses.py (already written and verified) exposes:

    adaptation_loss(task_loss, z_pre, z_post, pairs=None, weights=None,
                     lambda1=1.0, lambda2=1.0, lambda3=1.0,
                     distance="euclidean") -> (total_loss_tensor, components_dict)

    where components_dict has plain floats for "task"/"drift"/"cos"/"nbr"/"total".

Usage:
    python finetune_medclip.py \
        --split-json splits/exp_4/split.json \
        --exp-name exp_4 \
        --base-checkpoint checkpoints/base/model_weights.pt \
        --lambda1 1.0 --lambda2 1.0 --lambda3 1.0 \
        --epochs 5 --batch-size 16 --lr 1e-5 --seed 42

Naive baseline (no geometry regularization):
    python finetune_medclip.py --split-json splits/exp_4/split.json \
        --exp-name exp_4 --lambda1 0.0 --lambda2 0.0 --lambda3 0.0
"""

import argparse
import csv
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
from losses import adaptation_loss

EMBED_DIM = 512
NUM_CLASSES = 2
LABEL_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# split.json handling
# --------------------------------------------------------------------------

def load_paths(entries):
    """split.json entries may be plain path strings or dicts with a 'path'
    key. Either way, labels are inferred from the path itself later, never
    trusted from any label field that might also be present here."""
    paths = []
    for e in entries:
        if isinstance(e, str):
            paths.append(e)
        elif isinstance(e, dict):
            if "path" not in e:
                raise ValueError(f"split.json entry missing 'path' key: {e}")
            paths.append(e["path"])
        else:
            raise ValueError(f"Unrecognized split.json entry type: {type(e)}")
    return paths


def infer_label_from_path(path: str) -> str:
    parent_name = Path(path).parent.name.upper()
    if parent_name not in LABEL_TO_IDX:
        raise ValueError(
            f"Could not infer a NORMAL/PNEUMONIA label from path: {path} "
            f"(parent directory name was '{parent_name}')"
        )
    return parent_name


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class KermanySplitDataset(Dataset):
    def __init__(self, paths, processor: MedCLIPProcessor):
        self.paths = paths
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = LABEL_TO_IDX[infer_label_from_path(path)]
        img = Image.open(path).convert("L")
        pixel_values = self.processor(
            images=img, return_tensors="pt"
        )["pixel_values"].squeeze(0)
        return pixel_values, label, path


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def load_medclip(checkpoint_path: str, device: torch.device) -> MedCLIPModel:
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    state_dict = torch.load(checkpoint_path, map_location=device)
    # strict=True raises RuntimeError on any missing/unexpected key, so
    # reaching the line after this call already means 0 missing / 0 unexpected.
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    return model


# --------------------------------------------------------------------------
# Same-class pairs for L_nbr (Eq. 7)
# --------------------------------------------------------------------------

def build_same_class_pairs(labels):
    """labels: list/1D tensor of ints for the current batch. Returns
    (i, j) index pairs within the batch that share the same class --
    NORMAL-NORMAL or PNEUMONIA-PNEUMONIA only, by explicit design."""
    pairs = []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                pairs.append((i, j))
    return pairs


# --------------------------------------------------------------------------
# Evaluation on held-out d1 (never trained on)
# --------------------------------------------------------------------------

def evaluate(model, classifier, dataloader, device):
    model.eval()
    classifier.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for pixel_values, labels, _paths in tqdm(dataloader, desc="Evaluating on d1"):
            pixel_values = pixel_values.to(device)
            z = model.encode_image(pixel_values)
            logits = classifier(z)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(PNEUMONIA)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # e.g. d1 ended up single-class for some reason -- degenerate,
        # but should not crash a completed training run.
        auc = float("nan")
    return accuracy, f1, auc


# --------------------------------------------------------------------------
# Run bookkeeping
# --------------------------------------------------------------------------

def get_next_run_dir(exp_checkpoint_dir: Path) -> Path:
    """Auto-numbers run_001, run_002, ... Scoped to this exact exp's
    checkpoint directory, so it can't collide with another experiment's
    runs (unlike the known overwrite-check scoping bug in
    extract_embeddings.py, which looked for ANY z_*.npy in the folder)."""
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    existing_nums = []
    for d in exp_checkpoint_dir.iterdir():
        if d.is_dir() and d.name.startswith("run_") and d.name[len("run_"):].isdigit():
            existing_nums.append(int(d.name[len("run_"):]))
    next_num = max(existing_nums, default=0) + 1
    run_dir = exp_checkpoint_dir / f"run_{next_num:03d}"
    run_dir.mkdir(parents=False, exist_ok=False)  # fail loudly on any race
    return run_dir


def append_run_log(log_path: str, **row) -> None:
    row["timestamp"] = datetime.now(timezone.utc).isoformat()
    fieldnames = [
        "timestamp", "exp_name", "run_dir", "lambda1", "lambda2", "lambda3",
        "seed", "epochs", "batch_size", "lr", "distance",
        "accuracy", "f1", "auc", "checkpoint_path",
    ]
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-json", required=True, help="Path to this experiment's split.json (must have d1_prime/d1/meta keys)")
    p.add_argument("--exp-name", required=True, help="e.g. exp_1/exp_2/exp_3/exp_4 -- used for checkpoints/<exp-name>/")
    p.add_argument("--base-checkpoint", default="checkpoints/base/model_weights.pt")
    p.add_argument("--lambda1", type=float, required=True, help="Weight on L_drift (Eq. 5). 0.0 = naive baseline component.")
    p.add_argument("--lambda2", type=float, required=True, help="Weight on L_cos (Eq. 6).")
    p.add_argument("--lambda3", type=float, required=True, help="Weight on L_nbr (Eq. 7).")
    p.add_argument("--distance", default="euclidean", choices=["euclidean", "cosine"], help="Distance metric for L_nbr (Eq. 7)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda / cpu. Defaults to cuda if available, else cpu.")
    p.add_argument("--checkpoints-root", default="checkpoints")
    p.add_argument("--run-name", default=None, help="Override auto-numbered run_00X directory name")
    p.add_argument("--overwrite", action="store_true", help="Allow reusing an existing --run-name directory")
    p.add_argument("--runs-log", default="runs_log.csv")
    return p.parse_args()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[finetune_medclip] device = {device}")

    if not os.path.exists(args.base_checkpoint):
        raise FileNotFoundError(f"Base checkpoint not found: {args.base_checkpoint}")

    with open(args.split_json, "r") as f:
        split = json.load(f)
    for key in ("d1_prime", "d1", "meta"):
        if key not in split:
            raise KeyError(f"split.json at {args.split_json} is missing required key '{key}'")

    train_paths = load_paths(split["d1_prime"])
    eval_paths = load_paths(split["d1"])
    print(f"[finetune_medclip] d1_prime (train): {len(train_paths)} images")
    print(f"[finetune_medclip] d1 (held-out eval): {len(eval_paths)} images")

    processor = MedCLIPProcessor()
    train_loader = DataLoader(
        KermanySplitDataset(train_paths, processor),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    eval_loader = DataLoader(
        KermanySplitDataset(eval_paths, processor),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    print("[finetune_medclip] loading frozen (z_pre) model copy...")
    frozen_model = load_medclip(args.base_checkpoint, device)
    frozen_model.eval()
    for param in frozen_model.parameters():
        param.requires_grad_(False)

    print("[finetune_medclip] loading trainable (z_post) model copy...")
    trainable_model = load_medclip(args.base_checkpoint, device)
    trainable_model.train()

    classifier = nn.Linear(EMBED_DIM, NUM_CLASSES).to(device)
    ce_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(trainable_model.parameters()) + list(classifier.parameters()),
        lr=args.lr,
    )

    print(
        f"[finetune_medclip] lambda1={args.lambda1} lambda2={args.lambda2} "
        f"lambda3={args.lambda3} distance={args.distance} "
        f"(0/0/0 = naive fine-tuning baseline)"
    )

    for epoch in range(1, args.epochs + 1):
        trainable_model.train()
        classifier.train()
        running = {"task": 0.0, "drift": 0.0, "cos": 0.0, "nbr": 0.0, "total": 0.0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for pixel_values, labels, _paths in pbar:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                z_pre = frozen_model.encode_image(pixel_values)

            z_post = trainable_model.encode_image(pixel_values)
            logits = classifier(z_post)
            task_loss = ce_loss_fn(logits, labels)

            pairs = build_same_class_pairs(labels.tolist())

            total_loss, components = adaptation_loss(
                task_loss, z_pre, z_post,
                pairs=pairs, weights=None,
                lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
                distance=args.distance,
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            for k in running:
                running[k] += components[k]
            n_batches += 1
            pbar.set_postfix({k: f"{v / n_batches:.4f}" for k, v in running.items()})

        avg = {k: v / max(n_batches, 1) for k, v in running.items()}
        print(
            f"[finetune_medclip] epoch {epoch} avg -- task={avg['task']:.4f} "
            f"drift={avg['drift']:.4f} cos={avg['cos']:.4f} "
            f"nbr={avg['nbr']:.4f} total={avg['total']:.4f}"
        )

    print("[finetune_medclip] training complete -- evaluating once on held-out d1...")
    accuracy, f1, auc = evaluate(trainable_model, classifier, eval_loader, device)
    print(f"[finetune_medclip] d1 results -- accuracy={accuracy:.4f} f1={f1:.4f} auc={auc:.4f}")

    exp_checkpoint_dir = Path(args.checkpoints_root) / args.exp_name
    if args.run_name:
        run_dir = exp_checkpoint_dir / args.run_name
        if run_dir.exists() and not args.overwrite:
            raise FileExistsError(
                f"{run_dir} already exists. Pass --overwrite to reuse it, "
                f"or drop --run-name to auto-number a new run."
            )
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = get_next_run_dir(exp_checkpoint_dir)

    checkpoint_out_path = run_dir / "model_weights.pt"
    assert Path(args.base_checkpoint).resolve() != checkpoint_out_path.resolve(), \
        "Refusing to write to the base checkpoint -- checkpoints/base/ must never be overwritten."

    # Raw state_dict only, no optimizer state -- extract_embeddings.py loads
    # checkpoints via load_state_dict and expects exactly this format.
    torch.save(trainable_model.state_dict(), checkpoint_out_path)
    torch.save(classifier.state_dict(), run_dir / "classifier_head.pt")
    print(f"[finetune_medclip] saved checkpoint: {checkpoint_out_path}")

    append_run_log(
        args.runs_log,
        exp_name=args.exp_name,
        run_dir=str(run_dir),
        lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        distance=args.distance,
        accuracy=accuracy, f1=f1, auc=auc,
        checkpoint_path=str(checkpoint_out_path),
    )
    print(f"[finetune_medclip] appended run to {args.runs_log}")


if __name__ == "__main__":
    main()
