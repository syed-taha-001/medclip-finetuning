"""
filter_pre_to_holdout.py

Takes your existing full z_pre_embeddings.npy (100 images) and filters it
down to just the 20 held-out images, in the SAME ORDER as z_post_paths.npy,
so the two arrays are index-aligned for drift comparison.

Reads:
    ~/medclip_embeddings/z_pre_embeddings.npy   (full 100)
    ~/medclip_embeddings/holdout/z_post_paths.npy   (the 20 held-out paths, in order)

Saves:
    ~/medclip_embeddings/holdout/z_pre_embeddings.npy
    ~/medclip_embeddings/holdout/z_pre_labels.npy
    ~/medclip_embeddings/holdout/z_pre_paths.npy

Usage:
    python filter_pre_to_holdout.py
"""

import os
import numpy as np

FULL_DIR = os.path.expanduser('~/medclip_embeddings')
HOLDOUT_DIR = os.path.expanduser('~/medclip_embeddings/holdout')

full_embeddings = np.load(os.path.join(FULL_DIR, 'z_pre_embeddings.npy'))
full_labels = np.load(os.path.join(FULL_DIR, 'z_pre_labels.npy'))
full_paths = np.load(os.path.join(FULL_DIR, 'z_pre_paths.npy'))

holdout_paths = np.load(os.path.join(HOLDOUT_DIR, 'z_post_paths.npy'))

# Build a lookup so we can pull rows out of the full 100 in the exact
# order that z_post_paths.npy has them
path_to_row = {p: i for i, p in enumerate(full_paths)}
order = [path_to_row[p] for p in holdout_paths]

pre_embeddings_holdout = full_embeddings[order]
pre_labels_holdout = full_labels[order]
pre_paths_holdout = full_paths[order]

assert np.array_equal(pre_paths_holdout, holdout_paths), \
    "Alignment failed -- paths don't match z_post order."

np.save(os.path.join(HOLDOUT_DIR, 'z_pre_embeddings.npy'), pre_embeddings_holdout)
np.save(os.path.join(HOLDOUT_DIR, 'z_pre_labels.npy'), pre_labels_holdout)
np.save(os.path.join(HOLDOUT_DIR, 'z_pre_paths.npy'), pre_paths_holdout)

print(f"Filtered z_pre down to {len(order)} held-out images.")
print(f"Saved to {HOLDOUT_DIR}/ -- aligned with z_post.")
print("\nNext: python visualize_embeddings.py --emb_dir ~/medclip_embeddings/holdout")
