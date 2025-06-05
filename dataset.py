# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.View_Generator_Tube_Masking import ViewGenerator

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

class VideoFramesMultiViewDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                for img_name in sorted(os.listdir(folder_path)):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        self.image_paths.append(os.path.join(folder_path, img_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        tensor_img = transform(img)
        global_view, local_view1, local_view2 = ViewGenerator(tensor_img)

        return {
            'global_view': global_view,
            'local_view1': local_view1,
            'local_view2': local_view2
        }

def get_dataloader(root_dir, batch_size=2, num_workers=0):
    dataset = VideoFramesMultiViewDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader
