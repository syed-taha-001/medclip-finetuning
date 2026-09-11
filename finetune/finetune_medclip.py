"""
finetune_medclip.py

Naive fine-tuning of MedCLIP's vision encoder on a binary classification
task (NORMAL vs PNEUMONIA), with a stratified 80/20 train/held-out split.
This is the "naive" baseline in the proposal -- no geometry penalty yet
(that's eq. 8, for later once this baseline shows something worth fixing).

Saves:
    ~/medclip_finetuned/checkpoint.pt   -- fine-tuned model weights
    ~/medclip_finetuned/split.npy       -- dict: which paths were train vs held-out

Usage:
    python finetune_medclip.py
"""

import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

DATA_DIR = os.path.expanduser('~/chest_xray_data/chest_xray/train')
classes = ['NORMAL', 'PNEUMONIA']
LIMIT = 50          # per class, matches your z_pre extraction
TRAIN_FRAC = 0.8
EPOCHS = 5
LR = 1e-5
BATCH_SIZE = 16

OUT_DIR = os.path.expanduser('~/medclip_finetuned')
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Build the SAME 100-image list as extract_medclip_embeddings.py ----
image_paths, labels = [], []
for cls in classes:
    files = sorted(glob.glob(os.path.join(DATA_DIR, cls, '*.jpeg')))
    if LIMIT:
        files = files[:LIMIT]
    image_paths.extend(files)
    labels.extend([cls] * len(files))

label_to_idx = {c: i for i, c in enumerate(classes)}
label_ids = np.array([label_to_idx[l] for l in labels])
image_paths = np.array(image_paths)

# ---- Stratified 80/20 split: shuffle within each class, then split ----
train_idx, holdout_idx = [], []
for cls_id in range(len(classes)):
    cls_positions = np.where(label_ids == cls_id)[0]
    rng = np.random.RandomState(SEED)
    rng.shuffle(cls_positions)
    n_train = int(len(cls_positions) * TRAIN_FRAC)
    train_idx.extend(cls_positions[:n_train])
    holdout_idx.extend(cls_positions[n_train:])

train_idx = np.array(sorted(train_idx))
holdout_idx = np.array(sorted(holdout_idx))
print(f"Train: {len(train_idx)} images | Held-out: {len(holdout_idx)} images")

np.save(os.path.join(OUT_DIR, 'split.npy'), {
    'train_paths': image_paths[train_idx],
    'holdout_paths': image_paths[holdout_idx],
    'all_paths': image_paths,
})

# ---- Model + a small classification head on top of the vision encoder ----
processor = MedCLIPProcessor()
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()
model.to(device)

classifier_head = nn.Linear(512, len(classes)).to(device)

optimizer = torch.optim.Adam(
    list(model.vision_model.parameters()) + list(classifier_head.parameters()),
    lr=LR
)
loss_fn = nn.CrossEntropyLoss()

# ---- Training loop (train split only) ----
model.train()
train_paths = image_paths[train_idx]
train_labels = label_ids[train_idx]

for epoch in range(EPOCHS):
    perm = np.random.permutation(len(train_paths))
    epoch_loss = 0.0
    n_batches = 0

    for i in tqdm(range(0, len(train_paths), BATCH_SIZE), desc=f"Epoch {epoch+1}/{EPOCHS}"):
        batch_idx = perm[i:i + BATCH_SIZE]
        batch_paths = train_paths[batch_idx]
        batch_labels = torch.tensor(train_labels[batch_idx], dtype=torch.long).to(device)

        images = [Image.open(p).convert('RGB') for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)

        optimizer.zero_grad()
        img_embeds = model.encode_image(pixel_values)
        logits = classifier_head(img_embeds)
        loss = loss_fn(logits, batch_labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    print(f"Epoch {epoch+1}: avg loss = {epoch_loss / n_batches:.4f}")

# ---- Save fine-tuned checkpoint ----
torch.save(model.state_dict(), os.path.join(OUT_DIR, 'checkpoint.pt'))
print(f"\nSaved fine-tuned checkpoint to {OUT_DIR}/checkpoint.pt")
print(f"Saved split info to {OUT_DIR}/split.npy")
print("\nNext: run extract_post_embeddings.py to get z_post")
