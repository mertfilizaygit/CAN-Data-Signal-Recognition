import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

train_data = np.load('train_data.npy')


#num_rows = train_data.shape[0]
#print(num_rows)

# Perform Spectral Clustering on the train_data array
spectral_clustering = SpectralClustering(n_clusters=2, affinity='rbf', n_neighbors=10)
spectral_clustering.fit(train_data)

# Get the cluster labels for each sample in the train_data array
cluster_labels = spectral_clustering.labels_


cluster_counts = np.bincount(cluster_labels)
print("Number of samples in each cluster:", cluster_counts)


normal_signal_data = train_data[cluster_labels == np.argmax(cluster_counts)]
attacking_signal_data = train_data[cluster_labels == np.argmin(cluster_counts)]

np.save('normal_signal_data.npy', normal_signal_data)
np.save('attacking_signal_data.npy', attacking_signal_data)

# Plot the results
plt.scatter(train_data[:,0], train_data[:,1], c=cluster_labels)
plt.savefig('spectral_clustering.png')
