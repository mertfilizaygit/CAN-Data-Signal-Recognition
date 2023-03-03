import numpy as np
import pandas as pd
import matplotlib as plt
import can
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df_standardized = pd.read_csv("standardized_converted_data.csv")


pca = PCA()
pca.fit(df_standardized)
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
#This decision has been given according to the graph.
n_components = 3
pca = PCA(n_components=n_components)
pca.fit(df_standardized)
pca_transformed = pca.transform(df_standardized)
pca_transformed_df = pd.DataFrame(pca_transformed, columns=['Col1', 'Col2', 'Col3'])


#split the whole data into two, as train and test data
train_data, test_data = train_test_split(pca_transformed_df, test_size=0.2, shuffle=False)

#save them
np.save('train_data.npy', train_data)
np.save('test_data.npy', test_data)
