import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

train_data = np.load('train_data.npy')

# Perform Gaussian mixture clustering with 2 clusters
gmm = GaussianMixture(n_components=2)
gmm.fit(train_data)
cluster_labels = gmm.predict(train_data)

# Count the number of rows in each cluster
unique, counts = np.unique(cluster_labels, return_counts=True)
cluster_counts = dict(zip(unique, counts))

# Print the number of rows in each cluster
for cluster, count in cluster_counts.items():
    print("Cluster ", cluster, " has ", count, " rows.")


# Plot the results
#plt.scatter(train_data[:,0], train_data[:,1], c=cluster_labels)
#plt.title("Gaussian mixture clustering with 2 clusters")
#plt.xlabel("First Column")
#plt.ylabel("Second Column")

# Save the plot to your computer
#plt.savefig("gmm_clustering_plot.png")




