directory = "/Users/mertfilizay/Desktop/Praktikum TU Berlin/Data/Tracefiles 2/1"
dir_with_file = "/Users/mertfilizay/Desktop/Praktikum TU Berlin/Data/Tracefiles 2/1/NegativeTestsTrace_0x7E0.blf"

import numpy as np
import pandas as pd
import matplotlib as plt
import can
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


converted_data = []

def read_blf():


    files = os.listdir(directory)
    files.sort()

    # Open the CAN file with the 'rb' mode for reading in binary
    # rb means read binary
    for l in range(len(files)):
        if l == 0:
            continue
        can_file = open(os.path.join(directory,files[l]), "rb")
        bus = can.io.BLFReader(can_file)

        # Create a 'Bus' object using the file as a socket
        for msg in bus:
            data = msg.data
            converted_data.append(data)
read_blf()
df = pd.DataFrame(converted_data)

#Counts the number of rows in the accumulated data
print(df.shape[0])

df = pd.DataFrame(converted_data, columns=['Vehicle Speed', 'Braking Pressure', 'Yaw Rate','Steering Wheel Angle','Distance Totalizer','Battery Voltage','Engine Air Temperature','Engine RPM'])
#normalization_of_data
df_standardized = (df - df.mean())/df.std()
df_standardized.to_csv("standardized_converted_data.csv", index=False)






#plt.plot(pca.explained_variance_ratio_)
#plt.xlabel('Number of components')
#plt.ylabel('Explained variance ratio')
#plt.show()