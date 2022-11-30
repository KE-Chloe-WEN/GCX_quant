#!/usr/bin/env python
# coding: utf-8

# In[25]:


import os
import numpy as np
from czifile import CziFile
import matplotlib.pyplot as plt
from skimage import filters
from scipy.ndimage import gaussian_filter
import csv


# In[26]:


def get_czi_files():
    """This function returns all czi files under current folder as a list.
    """
    results = []
    files = os.scandir()
    for file in files:
        if file.is_file and file.name.endswith('czi'):
            results.append(file.name)
            
    return results

def drop_empty_dim(arr: np.array) -> np.array:
    '''
    This function is used for dropping empty dimension (shape = 1) of the input array.
    E.g., input array shape = [1, 1, 3, 4, 5] ==> [3, 4, 5] as output array shape
    
    '''
    
    dims_to_sq = np.where(np.array(arr.shape) == 1)[0]
    for dim_ in dims_to_sq[::-1]:
        arr = np.squeeze(arr, axis= dim_)
        
    return arr


# In[27]:


print('-'*20)
print(" - All channel analysis mode")
# print(" - Channel of czi image: {}\n".format(channel_index))
print('-'*20)
print(" Image processing in progress ...\n")

# Open czi files under current path.
n = 1
czi_files = get_czi_files() 
results = []
channel_index = 0 
folder_name = os.getcwd().split('/')[-1]

# Run through all czi files in the same folder.
for czi_file in czi_files:
    print(" - Processing image {} ... ".format(czi_file), end="")
    
    # Save all channel info into image_arrays array, 5d/6d array.
    with CziFile(czi_file) as czi:
        image_arrays = czi.asarray() 
    
    # Drop dimensions whose size is 1.
    image_arrays = drop_empty_dim(image_arrays)
    
    # Save defined channel to im.
    for channel_index in range(image_arrays.shape[0]):
        im = image_arrays[channel_index]
        
        # Get the size of the image.
        row, column = im.shape
        # Sum of intensity of all pixels.
        total = np.sum(im)
        # Calculate mean intensity.
        MFI = total / (row*column)
        
        # Save results.
        results.append({"id" : n, "file_name" : czi_file, "Mean Intensity" : MFI, "channel": channel_index + 1}) #dictionary, link index with a string...
        n += 1
    print(" Finished.")


# In[24]:


# Save results to csv file.
print("\n Analysis finished, results are saved to the current folder in {}_Con_Results.csv.".format(folder_name[:3]))
print('-'*20)
csv_file = "{}_Con_Results.csv".format(folder_name[:3])
csv_columns = ['id', 'file_name', 'Mean Intensity', 'channel']

try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(results)
except:
    print("Cannot save to csv file")


# In[ ]:




