import os, glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

processor = MedCLIPProcessor()
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()
model.to(device)
model.eval()

DATA_DIR = os.path.expanduser('~/chest_xray_data/chest_xray/train')
classes = ['NORMAL', 'PNEUMONIA']

# LIMIT caps how many images per class we process. Set to None for the full set.
LIMIT = 50

image_paths, labels = [], []
for cls in classes:
    files = sorted(glob.glob(os.path.join(DATA_DIR, cls, '*.jpeg')))
    if LIMIT:
        files = files[:LIMIT]
    image_paths.extend(files)
    labels.extend([cls] * len(files))

print(f"Found {len(image_paths)} images to process")

embeddings = []
batch_size = 16
with torch.no_grad():
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]
        images = [Image.open(p).convert('RGB') for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        img_embeds = model.encode_image(pixel_values)
        embeddings.append(img_embeds.cpu().numpy())

embeddings = np.concatenate(embeddings, axis=0)
labels = np.array(labels)
paths = np.array(image_paths)

os.makedirs(os.path.expanduser('~/medclip_embeddings'), exist_ok=True)
np.save(os.path.expanduser('~/medclip_embeddings/z_pre_embeddings.npy'), embeddings)
np.save(os.path.expanduser('~/medclip_embeddings/z_pre_labels.npy'), labels)
np.save(os.path.expanduser('~/medclip_embeddings/z_pre_paths.npy'), paths)

print("Embedding shape:", embeddings.shape)
print("Saved to ~/medclip_embeddings/")
