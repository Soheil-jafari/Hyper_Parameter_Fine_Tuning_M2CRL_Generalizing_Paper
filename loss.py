import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temp_student=0.1, temp_teacher=0.05):
        super().__init__()
        self.temp_student = temp_student
        self.temp_teacher = temp_teacher
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, student_features, teacher_features):
        student_out = student_features / self.temp_student
        teacher_out = (teacher_features / self.temp_teacher).detach()  # Stop gradient to teacher

        # Calculate InfoNCE style loss for each student embedding against all teacher embeddings in batch
        # For M2CRL, it's slightly different: p_t * log p_s (like DINO)
        # L_cl_gg = -p_t_g * log p_s_g
        # L_cl_gl = -p_t_g * log p_s_l

        # student_features: [B, C]
        # teacher_features: [B, C] (already detached in M2CRL logic)

        # Using the paper's formulation (cross-entropy based)
        # softmax(teacher_output / T_teacher) * log_softmax(student_output / T_student)

        student_log_probs = F.log_softmax(student_out, dim=-1)
        teacher_probs = F.softmax(teacher_out, dim=-1)  # Teacher output is detached, so probs are too

        loss = -torch.sum(teacher_probs * student_log_probs, dim=-1).mean()
        return loss


class ReconstructionLoss(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.criterion = nn.L1Loss(reduction='none')  # Or 'mean' if averaging over all masked pixels
        self.patch_size = patch_size

    def forward(self, reconstructed_patches_flat, target_image_views, bool_masked_patch_indices):


        B, C, H, W = target_image_views.shape
        pH = H // self.patch_size
        pW = W // self.patch_size
        N_total_patches = pH * pW

        # 1. Convert target_image_views to patches: [B, N_total_patches, P*P*C]
        target_patches_flat = target_image_views.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size,
                                                                                                    self.patch_size)
        target_patches_flat = target_patches_flat.permute(0, 2, 3, 1, 4, 5).contiguous()
        target_patches_flat = target_patches_flat.view(B, N_total_patches, -1)  # [B, N_total_patches, P*P*C]

        # 2. Calculate L1 loss for ALL patches
        loss_all_patches = self.criterion(reconstructed_patches_flat,
                                          target_patches_flat)  # [B, N_total_patches, P*P*C]
        loss_all_patches = loss_all_patches.mean(dim=-1)  # Average over P*P*C -> [B, N_total_patches]

        # 3. Select loss only for MASKED patches
        # bool_masked_patch_indices is [B, N_total_patches]
        masked_loss = loss_all_patches * bool_masked_patch_indices.float()  # Zero out loss for unmasked patches

        # 4. Average loss over the number of MASKED patches per image
        num_masked_patches_per_image = bool_masked_patch_indices.sum(dim=1).clamp(min=1e-9)  # [B]

        sum_loss_per_image_masked = masked_loss.sum(dim=1)  # [B]
        mean_loss_per_image_masked = sum_loss_per_image_masked / num_masked_patches_per_image  # [B]

        final_loss = mean_loss_per_image_masked.mean()  # Average over batch

        return final_loss