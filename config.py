import torch

config = {
    'model': {
        'backbone': 'vit_base_patch16_224',
        'image_size': 224,
        'patch_size': 16,
        'embed_dim': 768,
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
        'momentum': (0.9, 0.999),
        'mask_ratio': 0.9,
        'gamma': 0.6,
        'ema_momentum': 0.996,
        'temperature_student': 0.07,
        'temperature_teacher': 0.04,
    },
    'data': {
        'dataset_path': r'D:\detectron2\my_dataset\flatSubset',
        'num_workers': 0,
        'global_frames': 8,
        'num_local_views': 4,
        'max_frames_per_video_clip_input': 32
    },
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'fine_tune': {
        'pretrained_weights_path': r'C:\Users\Asus\PycharmProjects\PythonProject1\.venv\models\checkpoint.pth', # Keep your path
        'freeze_backbone': False
    },
    'eval': {
        'eval_frames_dir': r'D:\detectron2\my_dataset\test_video',
        'output_features_path': r'D:\detectron2\my_dataset\test_video\results\test_video_features.npy',
        'weights_path': r'models\trained_mmcrl.pth'
    }
}
