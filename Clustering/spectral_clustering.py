import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import os

# === CONFIG ===
features_path = r'D:\detectron2\my_dataset\test_video\results\test_video_features.npy'
n_clusters = 5  # Change this depending on how many segments you expect
output_labels_path = r'D:\detectron2\my_dataset\test_video\results\cluster_labels.npy'

# === LOAD FEATURES ===
features = np.load(features_path)
print(f"Loaded features with shape: {features.shape}")

# === NORMALIZATION (optional but helpful) ===
from sklearn.preprocessing import StandardScaler
features = StandardScaler().fit_transform(features)

# === SPECTRAL CLUSTERING ===
print("Running Spectral Clustering...")
sc = SpectralClustering(
    n_clusters=n_clusters,
    affinity='nearest_neighbors',  # or 'rbf'
    assign_labels='kmeans',
    random_state=42,
    n_neighbors=10
)
labels = sc.fit_predict(features)

# === SAVE LABELS ===
np.save(output_labels_path, labels)
print(f"Saved cluster labels to {output_labels_path}")

# === OPTIONAL: Visualize clustering (if using 2D reduction)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
features_2d = pca.fit_transform(features)

plt.figure(figsize=(10, 6))
plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='rainbow', s=5)
plt.title("Spectral Clustering Visualization")
plt.show()
