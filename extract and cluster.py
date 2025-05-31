# extract_features.py
import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from models.vit import VisionTransformer

# --- Config ---
data_dir = r"D:\\detectron2\\my_dataset\\flatSubset"
save_path = "features.npy"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model ---
model = VisionTransformer()
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval().to(device)

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# --- Extract CLS Features ---
features = []
frame_paths = []

for folder in sorted(os.listdir(data_dir)):
    folder_path = os.path.join(data_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith(".jpg") and not fname.endswith(".png"):
            continue
        img_path = os.path.join(folder_path, fname)
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            patch_tokens = model.patch_embed(tensor)
            cls_token = model.cls_token.expand(tensor.shape[0], -1, -1)
            tokens = torch.cat((cls_token, patch_tokens), dim=1)
            tokens = model.pos_drop(tokens + model.pos_embed[:, :tokens.size(1), :])
            for blk in model.blocks:
                tokens = blk(tokens)
            tokens = model.norm(tokens)
            cls_feat = tokens[:, 0]  # [B, C]

        features.append(cls_feat.cpu().numpy())
        frame_paths.append(img_path)

features = np.concatenate(features, axis=0)
np.save(save_path, features)
np.save("frame_paths.npy", frame_paths)
print(f"Saved features to {save_path} and frame paths to frame_paths.npy")
