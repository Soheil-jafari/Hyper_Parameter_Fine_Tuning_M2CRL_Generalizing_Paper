import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import numpy as np


class TubeMaskGenerator:
    def __init__(self, view_clip_size_thw, mask_ratio, tube_size_thw=(2, 8, 8)):
        self.T_view, self.H_view, self.W_view = view_clip_size_thw
        self.T_tube, self.H_tube, self.W_tube = tube_size_thw

        assert self.T_view % self.T_tube == 0, "View frames must be divisible by tube temporal size"
        assert self.H_view % self.H_tube == 0, "View height must be divisible by tube spatial height"
        assert self.W_view % self.W_tube == 0, "View width must be divisible by tube spatial width"

        self.num_tubes_t = self.T_view // self.T_tube
        self.num_tubes_h = self.H_view // self.H_tube
        self.num_tubes_w = self.W_view // self.W_tube
        self.num_tubes_total = self.num_tubes_t * self.num_tubes_h * self.num_tubes_w
        self.num_mask = int(mask_ratio * self.num_tubes_total)

    def __call__(self):  # Returns boolean mask: True if masked

        mask_flat = torch.zeros(self.num_tubes_total, dtype=torch.bool)
        if self.num_mask > 0 and self.num_mask <= self.num_tubes_total:  # Ensure num_mask is valid for randperm
            idx = torch.randperm(self.num_tubes_total)[:self.num_mask]
            mask_flat[idx] = True
        elif self.num_mask > self.num_tubes_total:  # Should not happen if mask_ratio <= 1
            mask_flat[:] = True  # Mask all if num_mask is too large

        # Reshape the mask.
        if self.num_tubes_t == 1:
            return mask_flat.view(self.num_tubes_h, self.num_tubes_w)
        else:
            return mask_flat.view(self.num_tubes_t, self.num_tubes_h, self.num_tubes_w)


class ViewGenerator:
    def __init__(self, image_size_hw=(224, 224),
                 global_frames=8, local_frames=4, num_local_views=2,
                 global_mask_ratio_rho=0.9, local_mask_ratio_rho=0.9,
                 tube_patch_size=16,  # Spatial patch size P for ViT
                 fagtm_gamma=0.6):
        self.image_size_hw = image_size_hw
        self.global_frames = global_frames
        self.local_frames = local_frames
        self.num_local_views = num_local_views

        self.base_frame_transform = T.Compose([
            T.Resize(self.image_size_hw, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])
        self.clip_augmentation_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.global_mask_ratio_rho = global_mask_ratio_rho
        self.local_mask_ratio_rho = local_mask_ratio_rho
        self.tube_patch_size = tube_patch_size
        self.fagtm_gamma = fagtm_gamma

        self.rtm_generator_local = TubeMaskGenerator(
            view_clip_size_thw=(1, self.image_size_hw[0], self.image_size_hw[1]),
            mask_ratio=self.local_mask_ratio_rho,
            tube_size_thw=(1, self.tube_patch_size, self.tube_patch_size)
        )

    def generate_views(self, video_frames_list_pil):
        num_total_frames = len(video_frames_list_pil)
        video_tensor_transformed = torch.stack([self.base_frame_transform(frame) for frame in video_frames_list_pil])

        global_view_clip = None
        if num_total_frames >= self.global_frames:
            start_idx_g = random.randint(0, num_total_frames - self.global_frames)
            global_view_clip_raw = video_tensor_transformed[start_idx_g: start_idx_g + self.global_frames]
            global_view_clip = torch.stack([self.clip_augmentation_transform(frame) for frame in global_view_clip_raw])
        else:
            indices = torch.arange(0, self.global_frames) % num_total_frames
            global_view_clip_raw = video_tensor_transformed[indices]
            global_view_clip = torch.stack([self.clip_augmentation_transform(frame) for frame in global_view_clip_raw])

        local_view_clips = []
        for _ in range(self.num_local_views):
            if num_total_frames >= self.local_frames:
                start_idx_l = random.randint(0, num_total_frames - self.local_frames)
                local_view_clip_raw = video_tensor_transformed[start_idx_l: start_idx_l + self.local_frames]
                local_view_clips.append(
                    torch.stack([self.clip_augmentation_transform(frame) for frame in local_view_clip_raw]))
            else:
                indices = torch.arange(0, self.local_frames) % num_total_frames
                local_view_clip_raw = video_tensor_transformed[indices]
                local_view_clips.append(
                    torch.stack([self.clip_augmentation_transform(frame) for frame in local_view_clip_raw]))
        return global_view_clip, local_view_clips

    def generate_rtm_spatial_mask_for_clip(self):
        return self.rtm_generator_local()

    def generate_fagtm_mask_from_agg_attention(self, aggregated_attention_map_spatial):
        num_total_spatial_patches = aggregated_attention_map_spatial.shape[0]
        num_patches_h = self.image_size_hw[0] // self.tube_patch_size
        num_patches_w = self.image_size_hw[1] // self.tube_patch_size
        assert num_total_spatial_patches == num_patches_h * num_patches_w, "Aggregated attention map size mismatch"

        num_visible_patches_N_v = round((1.0 - self.global_mask_ratio_rho) * num_total_spatial_patches)
        num_candidate_high_attention_N_h = int(np.ceil(self.fagtm_gamma * num_total_spatial_patches))

        sorted_indices = torch.argsort(aggregated_attention_map_spatial, descending=True)
        high_attention_candidate_indices = sorted_indices[:num_candidate_high_attention_N_h]

        actual_num_to_sample_visible = min(num_visible_patches_N_v, len(high_attention_candidate_indices))
        if len(high_attention_candidate_indices) == 0 and actual_num_to_sample_visible > 0:  # Edge case: no candidates but still need to make some visible
            visible_patch_indices = torch.randperm(num_total_spatial_patches,
                                                   device=aggregated_attention_map_spatial.device)[
                                    :actual_num_to_sample_visible]
        elif len(high_attention_candidate_indices) > 0:
            permuted_high_attention_indices = high_attention_candidate_indices[
                torch.randperm(len(high_attention_candidate_indices), device=aggregated_attention_map_spatial.device)]
            visible_patch_indices = permuted_high_attention_indices[:actual_num_to_sample_visible]
        else:  # No candidates and no visible patches needed
            visible_patch_indices = torch.tensor([], dtype=torch.long, device=aggregated_attention_map_spatial.device)

        fagtm_mask_flat_spatial = torch.ones(num_total_spatial_patches, dtype=torch.bool,
                                             device=aggregated_attention_map_spatial.device)
        if len(visible_patch_indices) > 0:
            fagtm_mask_flat_spatial[visible_patch_indices] = False

        return fagtm_mask_flat_spatial.view(num_patches_h, num_patches_w)