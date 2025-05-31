# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temp_student=0.07, temp_teacher=0.04):
        super().__init__()
        self.temp_student = temp_student
        self.temp_teacher = temp_teacher

    def forward(self, student_features, teacher_features):
        """
        student_features: [B, C]
        teacher_features: [B, C] (no grad)
        """
        student_logits = F.log_softmax(student_features / self.temp_student, dim=-1)
        teacher_probs = F.softmax((teacher_features / self.temp_teacher).detach(), dim=-1)
        loss = -torch.sum(teacher_probs * student_logits, dim=-1).mean()
        return loss

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, reconstructed, target, mask):
        """
        reconstructed: [B, N, C] — predicted by decoder
        target: [B, N, C] — original unmasked patches
        mask: [B, N, 1] — binary mask, 1 = masked
        """
        loss = self.criterion(reconstructed * mask, target * mask)
        return loss
