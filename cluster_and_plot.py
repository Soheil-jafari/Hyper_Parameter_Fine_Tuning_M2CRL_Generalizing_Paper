# cluster_and_plot.py
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from PIL import Image

# Config
features_path = 'path/to/saved/features.npy'  # same as config['eval']['output_features_path']
frames_dir = 'path/to/eval_frames_dir'         # same as config['eval']['eval_frames_dir']
num_clusters = 3

# Load features
features = np.load(features_path)

# KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)
labels = kmeans.labels_

# Visualize samples per cluster
frame_files = sorted(os.listdir(frames_dir))
clustered_frames = {i: [] for i in range(num_clusters)}
for label, fname in zip(labels, frame_files):
    clustered_frames[label].append(fname)

# Plot
for cluster_id, files in clustered_frames.items():
    plt.figure(figsize=(10, 2))
    for i, fname in enumerate(files[:10]):
        img = Image.open(os.path.join(frames_dir, fname)).convert('RGB')
        plt.subplot(1, 10, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(f"Cluster {cluster_id}")
    plt.tight_layout()
    plt.show()
