# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from config import config
from models.vit import VisionTransformer
from dataset import get_dataloader
from loss import ContrastiveLoss

class MMCRL(nn.Module):
    def __init__(self, backbone_cls):
        super().__init__()
        self.student = backbone_cls()
        self.teacher = backbone_cls()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.student(x)

    def update_teacher(self, momentum):
        for s_param, t_param in zip(self.student.parameters(), self.teacher.parameters()):
            t_param.data = momentum * t_param.data + (1. - momentum) * s_param.data

def train():
    print("Starting training...")
    device = config['device']
    model = MMCRL(VisionTransformer).to(device)
    dataloader = get_dataloader(
        config['data']['dataset_path'],
        config['training']['batch_size'],
        config['data']['num_workers']
    )

    optimizer = optim.AdamW(
        model.student.parameters(),
        lr=config['training']['learning_rate'],
        betas=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )

    contrastive_loss_fn = ContrastiveLoss(
        temp_student=config['training']['temperature_student'],
        temp_teacher=config['training']['temperature_teacher']
    )

    loss_log = []

    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            print("Loaded a batch")
            global_view = batch['global_view'].to(device)
            local_view1 = batch['local_view1'].to(device)
            local_view2 = batch['local_view2'].to(device)

            with torch.no_grad():
                teacher_output = model.teacher(global_view)[:, 0, :]  # CLS token only

            student_output1 = model.student(local_view1)[:, 0, :]
            student_output2 = model.student(local_view2)[:, 0, :]

            loss1 = contrastive_loss_fn(student_output1, teacher_output)
            loss2 = contrastive_loss_fn(student_output2, teacher_output)

            loss = (loss1 + loss2) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_teacher(config['training']['ema_momentum'])

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_log.append(avg_loss)
        print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.4f}")

    # Save loss log to CSV
    with open("loss_log.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "Loss"])
        for i, l in enumerate(loss_log):
            writer.writerow([i + 1, l])

if __name__ == '__main__':
    train()
