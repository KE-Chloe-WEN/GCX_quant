#!/usr/bin/env python
# coding: utf-8

# In[58]:


import os
import numpy as np
from czifile import CziFile
import matplotlib.pyplot as plt
from skimage import filters
from scipy.ndimage import gaussian_filter
import csv


# In[59]:


def get_czi_files():
    """ This function returns all czi files under current folder as a list.
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


# In[60]:


# List of median intensity value.
prof = {}

# Open Czi file under the current path.
czi_files = get_czi_files() 

# Run through all czi files in the folder.
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
        
        # Flip all orthogonal view images to x-z.
        row, col = im.shape
        if row > col:
            im = np.transpose(im)
        
        row, col = im.shape
        length = np.arange(0, col, 1, dtype=int)
        sgi = np.empty([col, 1])
        
        # Find median of the intensity along the x-axis, sgi is median intensity.
        for i in length:
            prof_dic = {}
            sgi[i] = np.median(im[:,i])
        
        prof_dic['median'] = sgi
        prof[czi_file.split(".")[0][:3]+'_'+ str(channel_index)] = prof_dic
        
        # Plot and save graph for median intensity versus x location.
        plt.plot(length, sgi)    
        plt.xlabel('length')
        plt.ylabel('Median Intensity')
        plt.xlim(0, col)
        plt.savefig("{}_{}_Con_MedianIntensity.png".format(czi_file.split(".")[0][:3], channel_index))
        plt.close()


# In[ ]:


# Save median intensity data if needed for further use.

# print("\n Analysis finished, results are saved to the current folder in {}_Con_Results.csv.".format(folder_name[:3]))
# print('-'*20)
# csv_file = "{}_Con_Results.csv".format(folder_name[:3])
# csv_columns = ['id', 'file_name', 'mean', 'std', 'otsu', 'channel']
# try:
#    with open(csv_file, 'w') as csvfile:
#        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
#        writer.writeheader()
#        writer.writerows(results)
# except:
#    print("Cannot saving to csv file")


# In[ ]:




