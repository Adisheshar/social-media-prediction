import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import KFold
from sklearn.datasets import make_blobs

# Generating synthetic data for testing
data, labels = make_blobs(n_samples=1000, centers=5, random_state=42)

# Step 1: Train an Autoencoder for feature extraction
autoencoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(data.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),  # Latent space
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(data.shape[1], activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(data, data, epochs=50, batch_size=256, validation_split=0.2)

# Step 2: Extract latent features
latent_features = autoencoder.predict(data)

# Step 3: Hyperparameter Tuning for K-Means
param_grid = [3, 5, 7, 9]  # Different cluster numbers to test
best_n_clusters = None
best_score = -1

for n_clusters in param_grid:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_features)
    score = silhouette_score(latent_features, cluster_labels)  # Compute manually
    
    print(f"Clusters: {n_clusters}, Silhouette Score: {score:.4f}")
    
    if score > best_score:  # Choose the best cluster count
        best_score = score
        best_n_clusters = n_clusters

print(f"Best number of clusters: {best_n_clusters}")

# Step 4: Apply K-Means clustering
kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
clusters = kmeans.fit_predict(latent_features)

# Step 5: Evaluate the Model
silhouette_avg = silhouette_score(latent_features, clusters)
print("Silhouette Score:", silhouette_avg)

true_labels = labels  # Use actual ground truth labels
ari_score = adjusted_rand_score(true_labels, clusters)
print(f"Adjusted Rand Index: {ari_score:.2f}")

# Compute reconstruction loss
reconstructed_data = autoencoder.predict(data)
mse_loss = np.mean(np.square(data - reconstructed_data))
print(f"Reconstruction Loss: {mse_loss:.4f}")

# Step 6: Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
silhouette_scores = []

for train_index, test_index in kf.split(latent_features):
    X_train, X_test = latent_features[train_index], latent_features[test_index]
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    kmeans.fit(X_train)
    clusters_pred = kmeans.predict(X_test)
    score = silhouette_score(X_test, clusters_pred)
    silhouette_scores.append(score)

avg_silhouette_score = np.mean(silhouette_scores)
print(f"Average Silhouette Score: {avg_silhouette_score:.4f}")
