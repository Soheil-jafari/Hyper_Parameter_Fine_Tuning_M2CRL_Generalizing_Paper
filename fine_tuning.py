import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import config  # Assuming config.py is in the same directory or accessible via PYTHONPATH
from models.vit import VisionTransformer  # Assuming models/vit.py is accessible


# --- Updated PolypDiag Classification Dataset Class ---
class PolypDiagClassificationDataset(Dataset):
    def __init__(self, root_dir, split_file_name, video_frames_dir_name="videos_frames_extracted", transform=None):
        """
        Args:
            root_dir (str): Root directory of the PolypDiag dataset (e.g., 'D:/.../PolypDiag/PolypDiag').
                            This directory should contain 'splits' and 'videos_frames_extracted' folders.
            split_file_name (str): Name of the split file (e.g., 'train.txt' or 'val.txt').
            video_frames_dir_name (str): Name of the directory containing extracted video frames.
                                         Defaults to "videos_frames_extracted".
            transform (callable, optional): Optional transform to be applied on a frame.
        """
        self.root_dir = root_dir
        self.split_file_path = os.path.join(root_dir, 'splits', split_file_name)
        self.video_frames_base_path = os.path.join(root_dir, video_frames_dir_name)
        self.transform = transform
        self.frames_per_video_clip = config['data']['global_frames']  # Using global_frames for consistency (8 frames)

        self.video_infos = []  # List of (video_folder_path, label)
        self._load_video_infos()

    def _load_video_infos(self):
        """
        Reads the split file and populates video_infos.
        Each line in the split file is expected to be 'video_filename.mp4,label'.
        """
        if not os.path.exists(self.split_file_path):
            raise FileNotFoundError(f"Split file not found: {self.split_file_path}")

        with open(self.split_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                video_filename_with_ext, label_str = line.rsplit(',', 1)  # Split by last comma

                # Remove .mp4 extension to get the folder name for extracted frames
                video_folder_name = os.path.splitext(video_filename_with_ext)[0]
                video_folder_path = os.path.join(self.video_frames_base_path, video_folder_name)

                if not os.path.isdir(video_folder_path):
                    print(f"Warning: Frames folder not found for video: {video_folder_path}. Skipping.")
                    continue

                self.video_infos.append({
                    'video_folder_path': video_folder_path,
                    'label': int(label_str)
                })

        if not self.video_infos:
            raise RuntimeError(f"No video information loaded from {self.split_file_path}. "
                               "Ensure split file format is 'video_filename.mp4,label' and frame folders exist.")

    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self, idx):
        video_info = self.video_infos[idx]
        video_folder_path = video_info['video_folder_path']
        label = video_info['label']

        available_frames = sorted(
            [f for f in os.listdir(video_folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

        if not available_frames:
            raise RuntimeError(f"No frames found in {video_folder_path}")

        # Sample 'self.frames_per_video_clip' frames
        # If fewer frames are available, repeat existing frames or pad with last frame
        if len(available_frames) >= self.frames_per_video_clip:
            # Randomly sample 'frames_per_video_clip' indices
            start_index = random.randint(0, len(available_frames) - self.frames_per_video_clip)
            selected_frame_names = available_frames[start_index: start_index + self.frames_per_video_clip]
        else:
            # If not enough frames, sample with replacement (or just use all and repeat the last one)
            selected_frame_names = random.sample(available_frames, len(available_frames))  # Sample all available
            while len(selected_frame_names) < self.frames_per_video_clip:
                selected_frame_names.append(selected_frame_names[-1])  # Pad with last frame if needed
            random.shuffle(selected_frame_names)  # Shuffle to make padding less obvious

        # Load frames and apply transform
        clip_frames = []
        for frame_name in selected_frame_names:
            frame_path = os.path.join(video_folder_path, frame_name)
            frame_image = Image.open(frame_path).convert('RGB')
            if self.transform:
                frame_image = self.transform(frame_image)
            clip_frames.append(frame_image)

        # Stack frames into a single tensor [T, C, H, W]
        clip_tensor = torch.stack(clip_frames)

        return clip_tensor, label

# --- Classification Model Class (remains the same as before) ---
class ClassificationModel(nn.Module):
    def __init__(self, backbone_cls, img_size, patch_size, embed_dim, num_classes):
        super().__init__()
        self.backbone = backbone_cls(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            use_decoder=False
        )
        self.classification_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x will be [B, T, C, H, W] for video clips
        # The VisionTransformer's forward method handles this input shape and returns [B, embed_dim]
        # by internally processing frames and aggregating CLS features.
        cls_features = self.backbone(x)
        logits = self.classification_head(cls_features)
        return logits


def train_classification():
    print("Starting supervised fine-tuning for Classification on PolypDiag...")
    device = torch.device(config['device'])

    # Data transformations for fine-tuning
    transform = transforms.Compose([
        transforms.Resize((config['model']['image_size'], config['model']['image_size']),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Instantiate the classification dataset and dataloaders
    train_dataset = PolypDiagClassificationDataset(
        root_dir=config['data']['dataset_path'],  # This path points to 'PolypDiag/PolypDiag'
        split_file_name='train.txt',
        transform=transform
    )
    val_dataset = PolypDiagClassificationDataset(
        root_dir=config['data']['dataset_path'],
        split_file_name='val.txt',  # Assuming a val.txt also exists
        transform=transform
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['fine_tune']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['fine_tune']['batch_size'],
        shuffle=False,  # No need to shuffle validation data
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=True
    )

    print(f"Detected {len(train_dataset)} training videos for classification.")
    print(f"Detected {len(val_dataset)} validation videos for classification.")
    print(f"Number of classes: {config['model']['num_classes']}")

    # Instantiate the classification model
    model = ClassificationModel(
        VisionTransformer,
        img_size=config['model']['image_size'],
        patch_size=config['model']['patch_size'],
        embed_dim=config['model']['embed_dim'],
        num_classes=config['model']['num_classes']
    ).to(device)

    best_val_accuracy = 0.0  # Initialize best validation accuracy to 0
    fine_tuned_checkpoint_path = config['fine_tune']['save_path']
    original_pretrained_path = config['fine_tune']['pretrained_weights_path']

    if os.path.exists(fine_tuned_checkpoint_path):
        print(f"Attempting to load fine-tuned checkpoint from {fine_tuned_checkpoint_path}")
        try:
            # Load the entire model state_dict
            checkpoint = torch.load(fine_tuned_checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint)
            print("Fine-tuned checkpoint loaded successfully. Resuming training from this state.")
            # If you saved best_val_accuracy in the checkpoint, you would load it here.
            # For now, we assume it just loads model weights and we'll track accuracy from current run.
            # Or you could parse your fine_tuning_val_log.csv to find the highest saved accuracy
            # and initialize best_val_accuracy with it to ensure saving only strictly better models.
            # For simplicity, we restart tracking best_val_accuracy from 0.0,
            # ensuring any new improvement gets saved.

        except Exception as e:
            print(f"Error loading fine-tuned checkpoint: {e}. Falling back to original pre-trained weights.")
            # If fine-tuned checkpoint fails, try loading the original pre-trained backbone
            if os.path.exists(original_pretrained_path):
                try:
                    pretrained_state_dict = torch.load(original_pretrained_path, map_location=device,
                                                       weights_only=False)
                    # Filter for backbone if the checkpoint contains other parts like decoder/projector
                    filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if
                                           not k.startswith('decoder.') and not k.startswith('projector.')}
                    model.backbone.load_state_dict(filtered_state_dict, strict=False)
                    print(f"Original pre-trained backbone weights from {original_pretrained_path} loaded successfully.")
                except Exception as e_pretrain:
                    print(
                        f"Error loading original pre-trained weights: {e_pretrain}. Proceeding with randomly initialized backbone.")
            else:
                print(
                    f"Original pre-trained weights not found at {original_pretrained_path}. Proceeding with randomly initialized backbone.")

    else:  # If fine-tuned checkpoint does not exist, load original pre-trained weights
        print(
            f"Fine-tuned checkpoint not found at {fine_tuned_checkpoint_path}. Attempting to load original pre-trained backbone weights from {original_pretrained_path}")
        if os.path.exists(original_pretrained_path):
            try:
                pretrained_state_dict = torch.load(original_pretrained_path, map_location=device, weights_only=False)
                # Filter for backbone if the checkpoint contains other parts like decoder/projector
                filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if
                                       not k.startswith('decoder.') and not k.startswith('projector.')}
                model.backbone.load_state_dict(filtered_state_dict, strict=False)
                print("Original pre-trained backbone weights loaded successfully.")
            except Exception as e:
                print(f"Error loading original pre-trained weights: {e}")
                print(
                    "Ensure 'config['fine_tune']['pretrained_weights_path']' points to the correct student encoder .pth file or a compatible model state_dict.")
                print("Proceeding with randomly initialized backbone. This will likely result in lower performance.")
        else:
            print(
                f"Original pre-trained weights not found at {original_pretrained_path}. Proceeding with randomly initialized backbone.")

    # Freeze backbone parameters if specified in config
    if config['fine_tune'].get('freeze_backbone', False):
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen for fine-tuning. Only classification head will be trained.")
    else:
        print("Backbone unfrozen for fine-tuning. Both backbone and classification head will be trained.")

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['fine_tune']['learning_rate'],
        momentum=config['fine_tune']['momentum'],
        weight_decay=config['training']['weight_decay']
    )
    criterion = nn.CrossEntropyLoss().to(device)

    train_loss_log = []
    val_loss_log = []

    print("Starting classification fine-tuning training loop...")
    best_val_accuracy = 0.0

    for epoch in range(config['fine_tune']['epochs']):
        # --- Training Phase ---
        model.train()
        epoch_train_loss = 0.0
        train_correct_predictions = 0
        train_total_predictions = 0

        for batch_idx, (video_clips, labels) in enumerate(train_dataloader):
            video_clips = video_clips.to(device, non_blocking=True)  # [B, T, C, H, W]
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(video_clips)  # Outputs logits [B, num_classes]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_total_predictions += labels.size(0)
            train_correct_predictions += (predicted == labels).sum().item()

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch + 1} Train Batch {batch_idx}/{len(train_dataloader)} Loss: {loss.item():.4f}")

        avg_train_loss = epoch_train_loss / len(train_dataloader)
        train_accuracy = 100 * train_correct_predictions / train_total_predictions
        train_loss_log.append([epoch + 1, avg_train_loss, train_accuracy])
        print(f"Epoch {epoch + 1} Training finished. Avg Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        # --- Validation Phase ---
        model.eval()  # Set model to evaluation mode
        epoch_val_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0

        with torch.no_grad():  # Disable gradient calculations for validation
            for batch_idx, (video_clips, labels) in enumerate(val_dataloader):
                video_clips = video_clips.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(video_clips)
                loss = criterion(outputs, labels)

                epoch_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total_predictions += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()

        avg_val_loss = epoch_val_loss / len(val_dataloader)
        val_accuracy = 100 * val_correct_predictions / val_total_predictions
        val_loss_log.append([epoch + 1, avg_val_loss, val_accuracy])
        print(f"Epoch {epoch + 1} Validation finished. Avg Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

        # Save best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), config['fine_tune']['save_path'])
            print(f"Saved best model with Val Accuracy: {best_val_accuracy:.2f}% to {config['fine_tune']['save_path']}")

    # Save fine-tuning logs to CSV
    with open("fine_tuning_train_log.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Train Accuracy"])
        writer.writerows(train_loss_log)
    print("Fine-tuning training log saved to fine_tuning_train_log.csv")

    with open("fine_tuning_val_log.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Validation Loss", "Validation Accuracy"])
        writer.writerows(val_loss_log)
    print("Fine-tuning validation log saved to fine_tuning_val_log.csv")


if __name__ == '__main__':
    train_classification()