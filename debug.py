# debug.py
import torch
from config import config
from dataset import get_dataloader
from models.vit import VisionTransformer
from models.projector import MLPProjector

if __name__ == '__main__':
    device = config['device']
    print("Loading dataloader...")
    dataloader = get_dataloader(
        config['data']['dataset_path'],
        batch_size=2,
        num_workers=0
    )

    model = VisionTransformer().to(device)
    projector = MLPProjector().to(device)

    print("Running one batch through ViT + Projector")
    for batch in dataloader:
        imgs = batch['image_tensor'].to(device)
        print("Input images:", imgs.shape)

        tokens = model(imgs)  # [B, N, C]
        print("Patch tokens:", tokens.shape)

        proj = projector(tokens)  # [B, N, proj_dim]
        print("Projected tokens:", proj.shape)

        break
