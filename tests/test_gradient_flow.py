"""
tests/test_gradient_flow.py

The real end-to-end proof that the training loop works, not just that the
loss math is correct on paper (that's what test_losses_vs_metrics.py and
test_nbr_empty_pairs.py already cover on synthetic data).

This test:
  1. Loads the real base MedCLIP checkpoint into TWO model instances:
     - a frozen copy (produces z_pre, no grad, .detach()'d)
     - a trainable copy (produces z_post, grad enabled)
  2. Runs a real batch of real Kermany images (mixed NORMAL/PNEUMONIA,
     preprocessed exactly like extract_embeddings.py: .convert("L") ->
     MedCLIPProcessor) through both.
  3. Attaches a small trainable classification head on top of z_post and
     computes a real cross-entropy task loss against the NORMAL/PNEUMONIA
     labels, so L_task in Eq. 8 is a genuine supervised loss, not a stand-in.
  4. Builds same-class-only neighbor pairs from the batch (per the standing
     design decision: L_nbr pairs are NORMAL-NORMAL / PNEUMONIA-PNEUMONIA
     only) and calls losses.adaptation_loss(...) — the actual function
     finetune_medclip.py will call.
  5. Calls .backward() and asserts the trainable model's vision-encoder
     parameters received real (non-None, non-zero, non-NaN, non-Inf)
     gradients, while the frozen copy's parameters received none.
  6. Takes one optimizer.step() and asserts at least one real weight
     actually changed value.

If this test passes, the combined loss genuinely backpropagates into the
real model end-to-end — proof the training loop works, not just the math.

Run directly (also usable outside pytest for quick manual debugging):
    python tests/test_gradient_flow.py

Or via pytest from repo root:
    pytest tests/test_gradient_flow.py -v -s
"""

import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# --------------------------------------------------------------------------- #
# Path setup — resolved relative to this file, not cwd, since this project
# has repeatedly hit "which directory am I actually in" bugs.
# --------------------------------------------------------------------------- #
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from losses import adaptation_loss  # noqa: E402  (path set up above)

try:
    from medclip import MedCLIPModel, MedCLIPProcessor, MedCLIPVisionModelViT
except ImportError as e:
    print(
        "ERROR: could not import medclip. Activate the 'medclip' conda env "
        f"before running this test.\n  {e}",
        file=sys.stderr,
    )
    sys.exit(1)


CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "base" / "model_weights.pt"
SPLIT_PATH = REPO_ROOT / "data" / "experiments" / "exp_4" / "split.json"
LABEL_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}
BATCH_SIZE_PER_CLASS = 4  # 4 NORMAL + 4 PNEUMONIA = 8 total, guarantees both
                          # classes present so same-class neighbor pairs exist


# --------------------------------------------------------------------------- #
# Minimal, self-contained data loading (deliberately NOT importing
# extract_embeddings.py's full loader — this test only needs a small fixed
# mixed-class batch, and keeping it self-contained means this test can't
# break just because extract_embeddings.py's loader changes shape later).
# --------------------------------------------------------------------------- #
def infer_label(rel_path: str) -> int:
    parent = Path(rel_path).parent.name.upper()
    if parent not in LABEL_TO_IDX:
        raise ValueError(f"Can't infer label from path '{rel_path}' (parent '{parent}')")
    return LABEL_TO_IDX[parent]


def build_mixed_batch():
    """Returns (pixel_values: (B,C,H,W) tensor, labels: (B,) LongTensor)."""
    with open(SPLIT_PATH, "r") as f:
        split = json.load(f)
    d1_paths = split["d1"]
    if d1_paths and isinstance(d1_paths[0], dict):
        d1_paths = [e.get("path", e.get("image", e.get("file"))) for e in d1_paths]

    by_class = {"NORMAL": [], "PNEUMONIA": []}
    for p in d1_paths:
        label_name = Path(p).parent.name.upper()
        if label_name in by_class and len(by_class[label_name]) < BATCH_SIZE_PER_CLASS:
            by_class[label_name].append(p)
        if all(len(v) >= BATCH_SIZE_PER_CLASS for v in by_class.values()):
            break

    chosen_paths = by_class["NORMAL"] + by_class["PNEUMONIA"]
    assert len(chosen_paths) == 2 * BATCH_SIZE_PER_CLASS, (
        f"Could not find {BATCH_SIZE_PER_CLASS} images of each class in "
        f"{SPLIT_PATH}'s d1 subset -- found {[(k, len(v)) for k, v in by_class.items()]}"
    )

    processor = MedCLIPProcessor()
    pixel_tensors, labels = [], []
    for rel_path in chosen_paths:
        img = Image.open(REPO_ROOT / rel_path).convert("L")
        processed = processor(images=img, return_tensors="pt")
        pixel_tensors.append(processed["pixel_values"].squeeze(0))
        labels.append(infer_label(rel_path))

    pixel_values = torch.stack(pixel_tensors, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return pixel_values, labels


def build_same_class_pairs(labels: torch.Tensor):
    """All (i, j) index pairs within the batch that share a label."""
    pairs = []
    n = labels.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i].item() == labels[j].item():
                pairs.append((i, j))
    return pairs


def load_model_copy():
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model


def get_image_embeddings(model, pixel_values):
    if hasattr(model, "encode_image"):
        return model.encode_image(pixel_values)
    feats = model.vision_model(pixel_values)
    if isinstance(feats, (tuple, list)):
        feats = feats[0]
    return model.visual_projection(feats)


# --------------------------------------------------------------------------- #
# The actual test
# --------------------------------------------------------------------------- #
def test_gradient_flow_real_model():
    assert CHECKPOINT_PATH.exists(), f"Checkpoint not found: {CHECKPOINT_PATH}"
    assert SPLIT_PATH.exists(), f"Split file not found: {SPLIT_PATH}"

    torch.manual_seed(0)

    pixel_values, labels = build_mixed_batch()
    pairs = build_same_class_pairs(labels)
    assert len(pairs) > 0, "No same-class pairs in this batch -- shouldn't happen with 4+4 split"

    # --- frozen copy: produces z_pre, no grad ---
    model_frozen = load_model_copy()
    model_frozen.eval()
    model_frozen.requires_grad_(False)
    with torch.no_grad():
        z_pre = get_image_embeddings(model_frozen, pixel_values)
    z_pre = z_pre.detach()
    assert z_pre.requires_grad is False, "z_pre must be detached (frozen base output)"

    # --- trainable copy: produces z_post, grad enabled ---
    model_trainable = load_model_copy()
    model_trainable.train()
    model_trainable.requires_grad_(True)
    z_post = get_image_embeddings(model_trainable, pixel_values)
    assert z_post.requires_grad is True, "z_post must require grad (trainable model output)"
    assert z_pre.shape == z_post.shape, f"z_pre {z_pre.shape} vs z_post {z_post.shape} mismatch"

    # --- real supervised task loss: small trainable head on z_post ---
    embed_dim = z_post.shape[-1]
    classifier_head = nn.Linear(embed_dim, 2)
    logits = classifier_head(z_post)
    task_loss = F.cross_entropy(logits, labels)

    # --- combined adaptation loss (Eq. 8), same call finetune_medclip.py makes ---
    total_loss, components = adaptation_loss(
        task_loss=task_loss,
        z_pre=z_pre,
        z_post=z_post,
        pairs=pairs,
        weights=None,
        lambda1=1.0,
        lambda2=1.0,
        lambda3=1.0,
        distance="euclidean",
    )
    print(f"[test_gradient_flow] loss components: {components}")
    assert total_loss.requires_grad is True
    for key in ("task", "drift", "cos", "nbr", "total"):
        val = components[key]
        assert val == val, f"{key} loss is NaN"          # NaN != NaN
        assert val not in (float("inf"), float("-inf")), f"{key} loss is Inf"

    # --- backward pass ---
    total_loss.backward()

    # --- check the TRAINABLE model's vision-encoder params got real gradients ---
    checked_any = False
    for name, param in model_trainable.vision_model.named_parameters():
        if param.grad is None:
            continue
        checked_any = True
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
        assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
    assert checked_any, (
        "No vision-encoder parameter received a gradient at all -- "
        "the loss is not actually connected to the model's weights."
    )
    grad_norm_sum = sum(
        p.grad.abs().sum().item()
        for p in model_trainable.vision_model.parameters()
        if p.grad is not None
    )
    assert grad_norm_sum > 0.0, "All vision-encoder gradients are exactly zero"
    print(f"[test_gradient_flow] sum of |grad| across vision encoder: {grad_norm_sum:.6f}")

    # --- check the FROZEN model's params got NO gradients ---
    for name, param in model_frozen.named_parameters():
        assert param.grad is None, f"Frozen model param '{name}' unexpectedly has a gradient"

    # --- optimizer step actually changes weights ---
    before = copy.deepcopy(
        next(model_trainable.vision_model.parameters()).detach().clone()
    )
    optimizer = torch.optim.SGD(
        list(model_trainable.parameters()) + list(classifier_head.parameters()),
        lr=0.01,
    )
    optimizer.step()
    after = next(model_trainable.vision_model.parameters()).detach().clone()
    assert not torch.equal(before, after), (
        "First vision-encoder parameter did not change after optimizer.step() "
        "-- gradients existed but the optimizer isn't actually updating weights."
    )
    changed = (before - after).abs().sum().item()
    print(f"[test_gradient_flow] first param total abs change after step: {changed:.6f}")

    print("[test_gradient_flow] PASSED: gradients flow end-to-end through the "
          "real model, and an optimizer step genuinely changes real weights.")


if __name__ == "__main__":
    test_gradient_flow_real_model()
