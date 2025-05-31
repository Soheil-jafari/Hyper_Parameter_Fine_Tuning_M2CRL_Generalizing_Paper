# STEP 1: CLEAN CONFIG AND TRAINING SKELETON FOR MMCRL

# config.py
config = {
    'model': {
        'backbone': 'vit_base_patch16_224',  # later you can change to resnet if you want
        'image_size': 224,
        'patch_size': 16,
        'embed_dim': 768,
    },
    'training': {
        'epochs': 6,
        'batch_size': 12,
        'learning_rate': 2e-5,
        'weight_decay': 4e-2,
        'warmup_epochs': 10,
        'momentum': (0.9, 0.999),
        'mask_ratio': 0.9,
        'gamma': 0.6,  # attention threshold
        'ema_momentum': 0.996,
        'temperature_student': 0.07,
        'temperature_teacher': 0.04,
    },
    'data': {
        'dataset_path': r'D:\detectron2\my_dataset\flatSubset',
        'num_workers': 0,
    },
    'device': 'cuda',
    'fine_tune': {
        'pretrained_weights_path': r'C:\Users\Asus\PycharmProjects\PythonProject1\.venv\models\checkpoint.pth',
        'freeze_backbone': False  # You can set to True if you want to freeze the encoder and train only projection head
    }

}

# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from config import config
from models.vit import VisionTransformer
from dataset import get_dataloader
from loss import ContrastiveLoss, ReconstructionLoss

class MMCRL(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.student = backbone()
        self.teacher = backbone()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.student(x)

    def update_teacher(self, momentum):
        for s_param, t_param in zip(self.student.parameters(), self.teacher.parameters()):
            t_param.data = momentum * t_param.data + (1. - momentum) * s_param.data


def train():
    device = config['device']
    model = MMCRL(VisionTransformer).to(device)
    dataloader = get_dataloader(config['data']['dataset_path'], config['training']['batch_size'])

    optimizer = optim.AdamW(
        model.student.parameters(),
        lr=config['training']['learning_rate'],
        betas=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )

    contrastive_loss = ContrastiveLoss(temp_student=config['training']['temperature_student'],
                                       temp_teacher=config['training']['temperature_teacher'])
    recon_loss = ReconstructionLoss()

    for epoch in range(config['training']['epochs']):
        model.train()
        for batch in dataloader:
            images = batch['image'].to(device)
            # TODO: Add view generation, masking, etc.

            output = model(images)
            # TODO: Compute contrastive and reconstruction loss

            loss = contrastive_loss(...) + recon_loss(...)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.update_teacher(config['training']['ema_momentum'])

        print(f"Epoch {epoch + 1} finished.")

config['eval'] = {
    'eval_frames_dir': r'D:\detectron2\my_dataset\test_video',
    'output_features_path': r'D:\detectron2\my_dataset\test_video\results\test_video_features.npy',
    'weights_path': r'models\trained_mmcrl.pth'  # update with your actual saved model path
}

if __name__ == '__main__':
    train()
