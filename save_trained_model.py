import torch
from models.vit import VisionTransformer
from train import MMCRL
from config import config

# Instantiate the model
model = MMCRL(VisionTransformer).to(config['device'])

# Load weights from RAM if still live
# If PyCharm kernel still has the trained model loaded, skip training again

# Save the model weights
torch.save(model.student.state_dict(), 'models/trained_mmcrl.pth')
print("Saved trained model to trained_mmcrl.pth")
