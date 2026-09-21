import json, numpy as np
from pathlib import Path
from PIL import Image
import torch
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor

repo_root = Path("..")
split = json.load(open("../data/experiments/exp_4/split.json"))
paths = split["d1"]

z = np.load("../embeddings/exp_4/base/z_d1.npy")

model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.load_state_dict(torch.load("../checkpoints/base/model_weights.pt", map_location="cpu"), strict=True)
model.eval()
proc = MedCLIPProcessor()

for idx in [0, len(paths)//2, len(paths)-1]:
    img = Image.open(repo_root / paths[idx]).convert("L")
    px = proc(images=img, return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        z_fresh = model.encode_image(px).squeeze(0).numpy()
    match = np.allclose(z_fresh, z[idx], atol=1e-5)
    print(idx, match)
