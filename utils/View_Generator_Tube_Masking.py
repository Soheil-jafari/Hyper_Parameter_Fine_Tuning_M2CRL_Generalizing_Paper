import torch
import torchvision.transforms as T
import random
import numpy as np
from einops import rearrange

class TubeMaskGenerator:
    def __init__(self, image_size, mask_ratio, tube_size=(2, 8, 8)):
        self.T, self.H, self.W = tube_size
        self.num_tubes = (image_size[0] // self.T) * (image_size[1] // self.H) * (image_size[2] // self.W)
        self.num_mask = int(mask_ratio * self.num_tubes)
        self.image_size = image_size
        self.tube_size = tube_size

    def __call__(self):
        mask = torch.zeros(self.num_tubes, dtype=torch.bool)
        idx = torch.randperm(self.num_tubes)[:self.num_mask]
        mask[idx] = True
        mask = mask.view(
            self.image_size[0] // self.T,
            self.image_size[1] // self.H,
            self.image_size[2] // self.W
        )
        return mask

class ViewGenerator:
    def __init__(self, transform, global_frames=8, local_frames=4, image_size=(8, 224, 224)):
        self.transform = transform
        self.global_frames = global_frames
        self.local_frames = local_frames
        self.image_size = image_size  # (T, H, W)

    def generate_views(self, video_tensor):
        # video_tensor shape: [T, C, H, W]
        assert video_tensor.shape[0] >= self.global_frames

        # Global view
        t_global_start = random.randint(0, video_tensor.shape[0] - self.global_frames)
        global_view = video_tensor[t_global_start:t_global_start + self.global_frames]

        # Local view
        t_local_start = random.randint(0, video_tensor.shape[0] - self.local_frames)
        local_view = video_tensor[t_local_start:t_local_start + self.local_frames]

        global_aug = torch.stack([self.transform(frame) for frame in global_view])
        local_aug = torch.stack([self.transform(frame) for frame in local_view])

        return global_aug, local_aug

    def generate_masks(self, method='rtm', attention_map=None):
        if method == 'rtm':
            mask_gen = TubeMaskGenerator(self.image_size, mask_ratio=0.6)
            return mask_gen()
        elif method == 'fagtm' and attention_map is not None:
            return self.fagtm_mask(attention_map)
        else:
            raise NotImplementedError("Unsupported masking method or missing attention map.")

    def fagtm_mask(self, attention_map):
        # attention_map: [T, H, W], higher means more important
        flat = attention_map.view(-1)
        _, indices = torch.topk(flat, int(flat.size(0) * 0.4), largest=False)  # mask least important
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask[indices] = True
        return mask.view(attention_map.shape)

# Example transform
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Sample usage
# video_tensor = torch.randn(16, 3, 256, 256)  # [T, C, H, W]
# vg = ViewGenerator(transform)
# g_view, l_view = vg.generate_views(video_tensor)
# mask = vg.generate_masks(method='rtm')