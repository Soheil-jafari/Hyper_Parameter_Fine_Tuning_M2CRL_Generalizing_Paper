import torch

config = {
    'model': {
        'backbone': 'vit_base_patch16_224',
        'image_size': 224,
        'patch_size': 16,
        'embed_dim': 768,
        'num_classes': 2, # For PolypDiag, assuming binary classification (polyp vs. normal)
        'decoder_embed_dim': 256,
        'decoder_depth': 1,
        'decoder_num_heads': 8,
        'projector_dim': 256
    },
    'training': {
        'epochs': 30,
        'batch_size': 12,
        'learning_rate': 2e-5,
        'weight_decay': 0.04,
        'warmup_epochs': 10,
        'momentum': (0.9, 0.999), # This is for pre-training AdamW betas
        'mask_ratio': 0.9,
        'gamma': 0.6,
        'ema_momentum': 0.996,
        'temperature_student': 0.07,
        'temperature_teacher': 0.04,
    },
    'data': {
    'dataset_path': r'D:\detectron2\my_dataset\PolypDia_dataset\PolypDiag\PolypDiag', # Updated path for PolypDiag
    'num_workers': 0, # Consider increasing this if you have many CPU cores and need faster data loading
    'global_frames': 8, # Keep this as 8, as the fine-tuning will use 8 frames per video
    'num_local_views': 4,
    'max_frames_per_video_clip_input': 32
    },
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'fine_tune': {
        'pretrained_weights_path': r'C:\Users\Asus\PycharmProjects\PythonProject1\.venv\models\checkpoint.pth', # Path to your saved student encoder weights
        'freeze_backbone': False, # Set to True to only train the classification head; False to fine-tune entire model
        'learning_rate': 1e-5, # As per paper for classification with SGD
        'momentum': 0.9, # As per paper for SGD optimizer
        'epochs': 20, # As per paper's mention for classification fine-tuning
        'batch_size': 4, # Changed to 4 as per paper's mention for PolypDiag classification
        'save_path': 'fineTuning_Weights/PolypDiag_fineTuning.pth' # Path to save the fine-tuned model
    },
    'eval': {
        'eval_frames_dir': r'D:\detectron2\my_dataset\test_video',
        'output_features_path': r'D:\detectron2\my_dataset\test_video\results\test_video_features.npy',
        'weights_path': r'models\checkpoint.pth' # This is likely for a general eval script, fine_tune uses its specific path
    },
}