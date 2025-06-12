import torch
import torch.nn as nn
import torch.optim as optim
import csv
import numpy as np
from config import config
from models.vit import VisionTransformer
from models.projector import MLPProjector
from dataset import get_dataloader  # Updated dataset
from loss import ContrastiveLoss, ReconstructionLoss
from utils.View_Generator_Tube_Masking import ViewGenerator


class MMCRL(nn.Module):
    def __init__(self, backbone_cls, projector_cls, img_size, patch_size, embed_dim=768, proj_dim=256,
                 use_decoder_for_student=True):  # Renamed for clarity
        super().__init__()
        self.student_encoder = backbone_cls(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            use_decoder=use_decoder_for_student,
            decoder_embed_dim=config['model'].get('decoder_embed_dim', 256),
            decoder_depth=config['model'].get('decoder_depth', 1),
            decoder_num_heads=config['model'].get('decoder_num_heads', 8)
        )
        self.student_projector = projector_cls(input_dim=embed_dim, proj_dim=proj_dim)

        self.teacher_encoder = backbone_cls(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            use_decoder=False
        )
        self.teacher_projector = projector_cls(input_dim=embed_dim, proj_dim=proj_dim)

        for param in self.teacher_encoder.parameters(): param.requires_grad = False
        for param in self.teacher_projector.parameters(): param.requires_grad = False

        student_encoder_state_dict = self.student_encoder.state_dict()
        teacher_encoder_state_dict_to_load = {
            k: v for k, v in student_encoder_state_dict.items()
            if not k.startswith('decoder.')
        }
        self.teacher_encoder.load_state_dict(teacher_encoder_state_dict_to_load)

        self.teacher_projector.load_state_dict(self.student_projector.state_dict())

    def forward_student(self, global_view_clip, local_view_clips, global_spatial_mask_bool_flat,
                        local_spatial_masks_bool_flat_list):
        # global_spatial_mask_bool_flat: [B, Np_g_spatial]
        # local_spatial_masks_bool_flat_list: list of [B, Np_l_spatial]

        B_g, T_g = global_view_clip.shape[0:2]
        global_frames_flat = global_view_clip.reshape(B_g * T_g, global_view_clip.shape[2], global_view_clip.shape[3],
                                                      global_view_clip.shape[4])

        global_mask_for_vit_student = global_spatial_mask_bool_flat.unsqueeze(1).repeat(1, T_g, 1).reshape(B_g * T_g,
                                                                                                           -1)

        student_global_cls_frames, reconstructed_global_frames = self.student_encoder(global_frames_flat,
                                                                                      bool_masked_pos=global_mask_for_vit_student)
        student_global_cls_agg = student_global_cls_frames.view(B_g, T_g, -1).mean(dim=1)
        student_global_cls_projected = self.student_projector(student_global_cls_agg)

        student_local_cls_projected_list = []
        reconstructed_local_clips_list = []

        for i, l_clip in enumerate(local_view_clips):
            B_l, T_l = l_clip.shape[0:2]
            local_frames_flat = l_clip.reshape(B_l * T_l, l_clip.shape[2], l_clip.shape[3], l_clip.shape[4])
            local_spatial_mask_flat = local_spatial_masks_bool_flat_list[i]  # [B, Np_l_spatial]
            local_mask_for_vit_student = local_spatial_mask_flat.unsqueeze(1).repeat(1, T_l, 1).reshape(B_l * T_l, -1)

            l_cls_frames, recon_l_frames = self.student_encoder(local_frames_flat,
                                                                bool_masked_pos=local_mask_for_vit_student)
            l_cls_agg = l_cls_frames.view(B_l, T_l, -1).mean(dim=1)
            l_cls_projected = self.student_projector(l_cls_agg)
            student_local_cls_projected_list.append(l_cls_projected)
            reconstructed_local_clips_list.append(recon_l_frames)

        return student_global_cls_projected, student_local_cls_projected_list, reconstructed_global_frames, reconstructed_local_clips_list

    def forward_teacher(self, global_view_clip):
        B_g, T_g = global_view_clip.shape[0:2]
        # For teacher, num_frames_in_clip is T_g to guide internal aggregation in ViT
        cls_features_agg, attention_map_agg = self.teacher_encoder(global_view_clip, is_teacher_for_fagtm=True,
                                                                   num_frames_in_clip=T_g)
        cls_projected_agg = self.teacher_projector(cls_features_agg)
        return cls_projected_agg, attention_map_agg

    @torch.no_grad()
    def update_teacher(self, momentum):
        for s_param_name, s_param_val in self.student_encoder.named_parameters():
            if not s_param_name.startswith('decoder.'):
                if s_param_name in self.teacher_encoder.state_dict():
                    t_param = self.teacher_encoder.state_dict()[s_param_name]
                    t_param.data.mul_(momentum).add_((1. - momentum) * s_param_val.data)

        for s_param_name, s_param_val in self.student_projector.named_parameters():
            if s_param_name in self.teacher_projector.state_dict():
                t_param = self.teacher_projector.state_dict()[s_param_name]
                t_param.data.mul_(momentum).add_((1. - momentum) * s_param_val.data)


def train():
    print("Starting M2CRL training (Video Data, Precise FAGTM)...")
    device = torch.device(config['device'])
    img_size_h, img_size_w = config['model']['image_size'], config['model']['image_size']
    patch_size = config['model']['patch_size']
    embed_dim = config['model']['embed_dim']
    proj_dim = config['model'].get('projector_dim', 256)

    global_frames_T_g = config['data'].get('global_frames', 8)
    local_frames_T_l = config['data'].get('local_frames', 4)
    num_local_views = config['data'].get('num_local_views', 2)
    max_frames_input = config['data'].get('max_frames_per_video_clip_input', 32)

    fagtm_gamma = config['training'].get('gamma', 0.6)
    global_mask_rho = config['training'].get('mask_ratio', 0.9)
    local_mask_rho = config['training'].get('mask_ratio', 0.9)

    view_gen_util_for_fagtm = ViewGenerator(
        image_size_hw=(img_size_h, img_size_w), global_frames=global_frames_T_g, local_frames=local_frames_T_l,
        num_local_views=num_local_views, global_mask_ratio_rho=global_mask_rho, local_mask_ratio_rho=local_mask_rho,
        tube_patch_size=patch_size, fagtm_gamma=fagtm_gamma
    )

    model = MMCRL(
        VisionTransformer, MLPProjector,
        img_size=img_size_h, patch_size=patch_size,
        embed_dim=embed_dim, proj_dim=proj_dim,
        use_decoder_for_student=True  # Explicitly state student uses decoder
    ).to(device)
    model.train()

    dataloader = get_dataloader(
        config['data']['dataset_path'], config['training']['batch_size'], config['data']['num_workers'],
        image_size_hw=(img_size_h, img_size_w), global_frames=global_frames_T_g, local_frames=local_frames_T_l,
        num_local_views=num_local_views, global_mask_rho=global_mask_rho, local_mask_rho=local_mask_rho,
        tube_patch_size=patch_size, fagtm_gamma=fagtm_gamma, max_frames_per_video_clip_input=max_frames_input
    )

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=config['training']['learning_rate'], betas=config['training']['momentum'],
                            weight_decay=config['training']['weight_decay'])
    contrastive_loss_fn = ContrastiveLoss(temp_student=config['training']['temperature_student'],
                                          temp_teacher=config['training']['temperature_teacher']).to(device)
    reconstruction_loss_fn = ReconstructionLoss(patch_size=patch_size).to(device)
    loss_log = []

    for epoch in range(config['training']['epochs']):
        epoch_total_loss, epoch_cl_loss, epoch_mvm_loss = 0.0, 0.0, 0.0
        for batch_idx, batch in enumerate(dataloader):
            global_view_clip_student_input = batch['global_view_clip'].to(device, non_blocking=True)
            local_view_clips_student_input = [lc.to(device, non_blocking=True) for lc in batch['local_view_clips']]
            local_rtm_spatial_masks_stacked = torch.stack(batch['local_rtm_spatial_masks']).to(device,
                                                                                               non_blocking=True)  # Stack for batch processing if possible

            local_rtm_masks_flat_list = []
            for i in range(num_local_views):

                mask_for_this_local_view_type = batch['local_rtm_spatial_masks'][i].to(device, non_blocking=True)
                local_rtm_masks_flat_list.append(
                    mask_for_this_local_view_type.reshape(mask_for_this_local_view_type.shape[0], -1))

            with torch.no_grad():
                teacher_global_cls_projected, teacher_agg_attention_map = model.forward_teacher(
                    global_view_clip_student_input)
                model.update_teacher(config['training']['ema_momentum'])

            batch_size_current = teacher_agg_attention_map.shape[0]
            global_mask_fagtm_spatial_list = []
            for i in range(batch_size_current):
                fagtm_spatial_mask = view_gen_util_for_fagtm.generate_fagtm_mask_from_agg_attention(
                    teacher_agg_attention_map[i])
                global_mask_fagtm_spatial_list.append(fagtm_spatial_mask)
            global_mask_student_fagtm_spatial_stacked = torch.stack(global_mask_fagtm_spatial_list)
            global_mask_student_fagtm_flat = global_mask_student_fagtm_spatial_stacked.reshape(batch_size_current, -1)

            student_global_cls_proj, student_local_cls_proj_list, recon_global_frames, recon_local_clips_frames = \
                model.forward_student(global_view_clip_student_input, local_view_clips_student_input,
                                      global_mask_student_fagtm_flat,
                                      local_rtm_masks_flat_list)  # Pass the list of flat masks

            loss_cl_gg = contrastive_loss_fn(student_global_cls_proj, teacher_global_cls_projected)
            loss_cl_gl = sum(contrastive_loss_fn(s_local_cls_p, teacher_global_cls_projected) for s_local_cls_p in
                             student_local_cls_proj_list)
            loss_cl_gl = loss_cl_gl / len(student_local_cls_proj_list) if student_local_cls_proj_list else 0.0
            total_contrastive_loss = loss_cl_gg + loss_cl_gl

            B_g, T_g_actual = global_view_clip_student_input.shape[0:2]
            target_global_frames_flat = global_view_clip_student_input.reshape(B_g * T_g_actual,
                                                                               global_view_clip_student_input.shape[2],
                                                                               img_size_h, img_size_w)
            global_mask_for_loss = global_mask_student_fagtm_flat.unsqueeze(1).repeat(1, T_g_actual, 1).reshape(
                B_g * T_g_actual, -1)
            loss_mvm_g = reconstruction_loss_fn(recon_global_frames, target_global_frames_flat, global_mask_for_loss)

            loss_mvm_l = 0.0
            if recon_local_clips_frames:
                for i, recon_local_frames_single_clip in enumerate(recon_local_clips_frames):
                    target_local_clip = local_view_clips_student_input[i]
                    B_l, T_l_actual = target_local_clip.shape[0:2]
                    mask_local_spatial_flat = local_rtm_masks_flat_list[i]  # Already [B, Np_flat]
                    mask_local_for_loss = mask_local_spatial_flat.unsqueeze(1).repeat(1, T_l_actual, 1).reshape(
                        B_l * T_l_actual, -1)
                    target_local_frames_flat = target_local_clip.reshape(B_l * T_l_actual, target_local_clip.shape[2],
                                                                         img_size_h, img_size_w)
                    loss_mvm_l += reconstruction_loss_fn(recon_local_frames_single_clip, target_local_frames_flat,
                                                         mask_local_for_loss)
                loss_mvm_l = loss_mvm_l / len(recon_local_clips_frames)
            total_reconstruction_loss = loss_mvm_g + loss_mvm_l

            total_loss = total_contrastive_loss + total_reconstruction_loss

            optimizer.zero_grad();
            total_loss.backward();
            optimizer.step()
            epoch_total_loss += total_loss.item()
            epoch_cl_loss += total_contrastive_loss.item() if isinstance(total_contrastive_loss,
                                                                         torch.Tensor) else float(
                total_contrastive_loss)
            epoch_mvm_loss += total_reconstruction_loss.item() if isinstance(total_reconstruction_loss,
                                                                             torch.Tensor) else float(
                total_reconstruction_loss)

            if batch_idx % 20 == 0:
                cl_item = total_contrastive_loss.item() if isinstance(total_contrastive_loss, torch.Tensor) else float(
                    total_contrastive_loss)
                mvm_item = total_reconstruction_loss.item() if isinstance(total_reconstruction_loss,
                                                                          torch.Tensor) else float(
                    total_reconstruction_loss)
                print(
                    f"E{epoch + 1} B{batch_idx}/{len(dataloader)} Loss:{total_loss.item():.4f} (CL:{cl_item:.4f}, MVM:{mvm_item:.4f})")

        avg_epoch_loss = epoch_total_loss / len(dataloader);
        avg_cl_loss = epoch_cl_loss / len(dataloader);
        avg_mvm_loss = epoch_mvm_loss / len(dataloader)
        loss_log.append([epoch + 1, avg_epoch_loss, avg_cl_loss, avg_mvm_loss])
        print(f"E{epoch + 1} Avg Loss:{avg_epoch_loss:.4f} (CL:{avg_cl_loss:.4f}, MVM:{avg_mvm_loss:.4f})")

    with open("m2crl_video_fagtm_loss_log.csv", "w", newline='') as f:
        writer = csv.writer(f);
        writer.writerow(["Epoch", "Total", "CL", "MVM"]);
        writer.writerows(loss_log)
    torch.save(model.student_encoder.state_dict(), 'm2crl_video_fagtm_student_encoder.pth')
    torch.save(model.student_projector.state_dict(), 'm2crl_video_fagtm_student_projector.pth')
    print("Saved trained student encoder and projector.")


if __name__ == '__main__':
    train()