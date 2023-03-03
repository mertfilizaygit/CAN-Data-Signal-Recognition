import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


train_data = np.load('train_data.npy')

# Assuming your data is stored in a numpy array called "data"
kmeans = KMeans(n_clusters=2) # initialize KMeans with number of clusters = 2
kmeans.fit(train_data) # fit the KMeans model to the data

labels = kmeans.predict(train_data)

# Get the cluster assignments for each sample
cluster_assignments = kmeans.labels_

# Count the number of samples in each cluster
cluster_counts = np.bincount(cluster_assignments)

print("Number of samples in each cluster:", cluster_counts)

# Plot the data points colored by their cluster assignment
#plt.scatter(train_data[:, 0], train_data[:, 1], c=labels)

# Add labels for the cluster centers
#cluster_centers = kmeans.cluster_centers_
#plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=200, linewidths=3, color='r')

# Show the plot
#plt.show()

#save the plot
#plt.savefig('k_means_clustering.png')



