# models/projector.py
import torch
import torch.nn as nn

class MLPProjector(nn.Module):
    def __init__(self, input_dim=768, proj_dim=256, hidden_dim=2048):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x):
        """
        x: [B, N, C] — token features from ViT
        Returns: [B, N, proj_dim]
        """
        return self.projector(x)
