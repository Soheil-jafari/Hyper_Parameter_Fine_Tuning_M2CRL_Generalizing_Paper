# utils/augment.py
import random
import torchvision.transforms as T
from PIL import Image

class ViewGenerator:
    def __init__(self, global_crop_size=(224, 224), local_crop_size=(96, 96)):
        self.global_crop = T.Compose([
            T.RandomResizedCrop(global_crop_size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.local_crop = T.Compose([
            T.RandomResizedCrop(local_crop_size, scale=(0.1, 0.5)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __call__(self, image: Image.Image):
        views = {
            'global_view': self.global_crop(image),
            'local_view1': self.local_crop(image),
            'local_view2': self.local_crop(image),
        }
        return views
