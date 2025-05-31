# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils.augment import ViewGenerator
import torchvision.transforms as T

class VideoFramesMultiViewDataset(Dataset):
    def __init__(self, root_dir, img_size=224):
        self.root_dir = root_dir
        self.image_paths = []

        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                for img_name in sorted(os.listdir(folder_path)):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        self.image_paths.append(os.path.join(folder_path, img_name))

        self.view_gen = ViewGenerator(global_crop_size=(img_size, img_size), local_crop_size=(96, 96))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        views = self.view_gen(image)  # returns dict of 3 views

        return {
            'global_view': views['global_view'],
            'local_view1': views['local_view1'],
            'local_view2': views['local_view2']
        }

def get_dataloader(root_dir, batch_size=2, num_workers=2):
    dataset = VideoFramesMultiViewDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    print("DataLoader created with", len(dataloader), "batches")
    return dataloader
