"""
extract_post_embeddings.py

Extracts embeddings from the FINE-TUNED checkpoint, using only the
held-out 20 images (never seen during fine-tuning). This is the
generalization test -- did fine-tuning distort the embedding space
even for images the model wasn't trained on?

Reads:
    ~/medclip_finetuned/checkpoint.pt
    ~/medclip_finetuned/split.npy   (holdout_paths)

Saves (to a NEW folder, separate from your full 100-image z_pre):
    ~/medclip_embeddings/holdout/z_post_embeddings.npy
    ~/medclip_embeddings/holdout/z_post_labels.npy
    ~/medclip_embeddings/holdout/z_post_paths.npy

Usage:
    python extract_post_embeddings.py
"""

import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

FT_DIR = os.path.expanduser('~/medclip_finetuned')
OUT_DIR = os.path.expanduser('~/medclip_embeddings/holdout')
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load held-out paths from the split saved during fine-tuning ----
split = np.load(os.path.join(FT_DIR, 'split.npy'), allow_pickle=True).item()
holdout_paths = split['holdout_paths']
print(f"Found {len(holdout_paths)} held-out images")

# Labels come from the folder name in each path (NORMAL / PNEUMONIA)
holdout_labels = np.array([
    'NORMAL' if 'NORMAL' in p else 'PNEUMONIA' for p in holdout_paths
])

# ---- Load the fine-tuned model ----
processor = MedCLIPProcessor()
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()  # base weights first
model.load_state_dict(torch.load(os.path.join(FT_DIR, 'checkpoint.pt'),
                                  map_location=device))
model.to(device)
model.eval()

# ---- Extract, same pattern as extract_medclip_embeddings.py ----
embeddings = []
batch_size = 16
with torch.no_grad():
    for i in tqdm(range(0, len(holdout_paths), batch_size)):
        batch_paths = holdout_paths[i:i + batch_size]
        images = [Image.open(p).convert('RGB') for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        img_embeds = model.encode_image(pixel_values)
        embeddings.append(img_embeds.cpu().numpy())

embeddings = np.concatenate(embeddings, axis=0)

np.save(os.path.join(OUT_DIR, 'z_post_embeddings.npy'), embeddings)
np.save(os.path.join(OUT_DIR, 'z_post_labels.npy'), holdout_labels)
np.save(os.path.join(OUT_DIR, 'z_post_paths.npy'), holdout_paths)

print(f"\nz_post shape: {embeddings.shape}")
print(f"Saved to {OUT_DIR}/")
print("\nNext: run filter_pre_to_holdout.py to build the matching z_pre subset")
