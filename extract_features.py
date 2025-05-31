# extract_features.py
import os
import torch
from torchvision import transforms
from PIL import Image
from models.vit import VisionTransformer
from config import config
import numpy as np
from tqdm import tqdm

# Setup
model = VisionTransformer().to(config['device'])
model.load_state_dict(torch.load(config['eval']['weights_path'], map_location=config['device']))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Directories
input_dir = config['eval']['eval_frames_dir']  # path to your folder of video frames
output_path = config['eval']['output_features_path']
features = []

# Read and process images
print("Extracting features...")
with torch.no_grad():
    for fname in tqdm(sorted(os.listdir(input_dir))):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue  # skip non-image files/folders
        path = os.path.join(input_dir, fname)
        img = transform(Image.open(path).convert('RGB')).unsqueeze(0).to(config['device'])
        tokens = model(img)  # [1, N+1, C], includes CLS token
        cls_token = tokens[:, 0, :]  # [1, C]
        features.append(cls_token.squeeze(0).cpu().numpy())


# Save all CLS tokens
np.save(output_path, np.array(features))
print(f"Saved extracted features to {output_path}")