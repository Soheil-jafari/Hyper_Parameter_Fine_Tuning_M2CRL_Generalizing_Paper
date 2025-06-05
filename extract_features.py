# extract_features.py
import os
import torch
from torchvision import transforms
from PIL import Image
from models.vit import VisionTransformer
from config import config
import numpy as np
from tqdm import tqdm

model = VisionTransformer().to(config['device'])
model.load_state_dict(torch.load(config['eval']['weights_path'], map_location=config['device']))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

input_dir = config['eval']['eval_frames_dir']
output_path = config['eval']['output_features_path']
features = []

print("Extracting features...")
with torch.no_grad():
    for fname in tqdm(sorted(os.listdir(input_dir))):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        path = os.path.join(input_dir, fname)
        img = transform(Image.open(path).convert('RGB')).unsqueeze(0).to(config['device'])
        tokens = model(img)
        cls_token = tokens[:, 0, :]
        features.append(cls_token.squeeze(0).cpu().numpy())

np.save(output_path, np.array(features))
print(f"Saved features to {output_path}")
