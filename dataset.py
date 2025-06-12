import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # No 'as T' to avoid conflict
from utils.View_Generator_Tube_Masking import ViewGenerator


class VideoClipDataset(Dataset):
    def __init__(self, root_dir, view_generator_instance, max_frames_per_video_clip=32):  # e.g. T_in
        self.root_dir = root_dir
        self.view_generator = view_generator_instance
        self.max_frames_per_video_clip = max_frames_per_video_clip

        self.video_folders = []
        self.frames_in_video = {}  # Store paths to frames for each video folder

        for video_folder_name in sorted(os.listdir(root_dir)):
            video_folder_path = os.path.join(root_dir, video_folder_name)
            if os.path.isdir(video_folder_path):
                frames = []
                for img_name in sorted(os.listdir(video_folder_path)):
                    if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                        frames.append(os.path.join(video_folder_path, img_name))
                if frames:
                    self.video_folders.append(video_folder_name)
                    self.frames_in_video[video_folder_name] = frames

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_folder_name = self.video_folders[idx]
        frame_paths = self.frames_in_video[video_folder_name]

        selected_frame_paths = frame_paths[:self.max_frames_per_video_clip]
        if len(selected_frame_paths) < self.max_frames_per_video_clip and len(
                selected_frame_paths) > 0:  # Pad if too short
            num_padding = self.max_frames_per_video_clip - len(selected_frame_paths)
            padding_paths = [selected_frame_paths[-1]] * num_padding  # Pad with last frame
            selected_frame_paths.extend(padding_paths)

        video_frames_pil = [Image.open(p).convert('RGB') for p in selected_frame_paths]

        # Generate augmented video clips (global [T_g,C,H,W], list of local [T_l,C,H,W])
        global_view_clip, local_view_clips = self.view_generator.generate_views(video_frames_pil)

        local_rtm_spatial_masks = [self.view_generator.generate_rtm_spatial_mask_for_clip() for _ in local_view_clips]


        return {
            'global_view_clip': global_view_clip,  # [T_g, C, H, W]
            'local_view_clips': local_view_clips,  # List of [T_l, C, H, W]
            'local_rtm_spatial_masks': local_rtm_spatial_masks,  # List of [Hp, Wp]
            'global_view_frames_count': global_view_clip.shape[0] if global_view_clip is not None else 0,  # T_g
            'local_view_frames_counts': [lc.shape[0] for lc in local_view_clips]  # list of T_l
        }


def get_dataloader(root_dir, batch_size=2, num_workers=0,
                   image_size_hw=(224, 224), global_frames=8, local_frames=4, num_local_views=2,
                   global_mask_rho=0.9, local_mask_rho=0.9, tube_patch_size=16, fagtm_gamma=0.6,
                   max_frames_per_video_clip_input=32):
    view_gen_instance = ViewGenerator(
        image_size_hw=image_size_hw, global_frames=global_frames, local_frames=local_frames,
        num_local_views=num_local_views, global_mask_ratio_rho=global_mask_rho,
        local_mask_ratio_rho=local_mask_rho, tube_patch_size=tube_patch_size, fagtm_gamma=fagtm_gamma
    )

    dataset = VideoClipDataset(root_dir, view_gen_instance, max_frames_per_video_clip=max_frames_per_video_clip_input)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
                            drop_last=True)
    return dataloader