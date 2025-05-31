import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# === CONFIG ===
frames_dir = r'D:\detectron2\my_dataset\test_video'  # where your frame images are
labels_path = r'D:\detectron2\my_dataset\test_video\results\cluster_labels.npy'

# === LOAD CLUSTER LABELS ===
labels = np.load(labels_path)
num_clusters = len(set(labels))

# === LOAD ALL FRAME FILES ===
frame_files = sorted([
    f for f in os.listdir(frames_dir)
    if f.lower().endswith(('.jpg', '.png', '.jpeg'))
])

assert len(frame_files) == len(labels), "Mismatch between frames and labels!"

# === ORGANIZE FRAMES BY CLUSTER ===
clustered_frames = {i: [] for i in range(num_clusters)}
for label, fname in zip(labels, frame_files):
    clustered_frames[label].append(fname)

# === PLOT 10 RANDOM FRAMES PER CLUSTER ===
for cluster_id in range(num_clusters):
    samples = random.sample(clustered_frames[cluster_id], min(10, len(clustered_frames[cluster_id])))

    plt.figure(figsize=(15, 2))
    for i, fname in enumerate(samples):
        img = Image.open(os.path.join(frames_dir, fname)).convert('RGB')
        plt.subplot(1, 10, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(f"Cluster {cluster_id}", fontsize=14)
    plt.tight_layout()
    plt.show()
