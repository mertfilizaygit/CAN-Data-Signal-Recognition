import os
import pandas as pd

directory = '/Users/mertfilizay/Desktop/Python_files/pythonProject1/Final Outputs'

#print(os.listdir(directory))

all_data = []

# loop through each file in the directory
for filename in os.listdir(directory):
    # check if the file is an Excel file and starts with "output"
    if filename.endswith('.xlsx') and filename.startswith('output'):
        # read the Excel file into a pandas dataframe
        df = pd.read_excel(os.path.join(directory, filename))
        # add the dataframe to the list
        all_data.append(df)

combined_data = pd.concat(all_data)
combined_data.to_excel(os.path.join(directory, 'combined_output_final_version.xlsx'), index=False)

