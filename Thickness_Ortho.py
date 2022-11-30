#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from czifile import CziFile
import matplotlib.pyplot as plt
from skimage import filters
from scipy.ndimage import gaussian_filter
import csv

def get_czi_files():
    """ This function returns all czi files under current folder as a list.
    """
    results = []
    files = os.scandir()
    for file in files:
        if file.is_file and file.name.endswith('czi'):
            results.append(file.name)
            
    return results


def cutout_blank(arr):
    return arr [:, arr.sum(axis = 0) !=0]

czi_files = get_czi_files() # czi files under current path.
k = 50  # Number of lines
simga_gauss = 0.5 # Standard deviation for Gaussian kernel
#channel_index = 0 # Define the channel index will be extracted.
folder_name = os.getcwd().split('/')[-1]
n = 1
results = [] # Result list

print('-'*20)
print(" Params used:\n")
print(" - Number of random lines used: {}".format(k))
print(" - Gaussian blur: {}".format(simga_gauss))
print(" - All channel analysis mode")
#print(" - Channel of czi image: {}\n".format(channel_index))
print('-'*20)
print(" Image processing in progress ...\n")
# Looping through all czi files.
for czi_file in czi_files:
    print(" - Processing image {} ... ".format(czi_file), end="")
    # Open czi_file and save all channel info into image_arrays array.
    with CziFile(czi_file) as czi:
        image_arrays = czi.asarray() # 5d/6d array
        
    for channel_index in range(image_arrays.shape[1]):
    # Save defined channel to im and squeeze to a 2D array.
        im = image_arrays[0][channel_index].squeeze(axis=-1) #2d array without last number
        
        row, col = im.shape
        if row > col:
           im = np.transpose(im)

        # Apply gaussian blur to image. 
        im_clean = gaussian_filter(im, sigma = 0.5) 
        otsu_threshold = filters.threshold_otsu(im_clean)
        im_clean = im_clean * (im_clean > otsu_threshold)

        im_clean = cutout_blank(im_clean) #0 summation of vertical, 1 is summation of horizontal

        rand_index = np.random.randint(im_clean.shape[1], size = k)
        #im_plot = im_clean.copy()
        #im_plot[:, rand_index] = np.max(im_plot) + 1000

        # im_clean[:, rand_index] = np.max(im_clean) 
        #plt.imsave('{}-{}.png'.format(czi_file.split('.')[0], channel_index), im_plot, cmap='gray')

        # Average thickness
        rand_thick = np.sum(im_clean[:, rand_index] !=0, axis=0) * 9.17 / 130 # in microns
        thick_mean = np.average(rand_thick)
        thick_std = np.std(rand_thick)
        
        results.append({"id" : n, "file_name" : czi_file, "mean" : thick_mean, "std" : thick_std, "otsu" : otsu_threshold, "channel": channel_index + 1}) #dictionary, link index with a string...
        n += 1
    print(" Finished.")

print("\n Analysis finished, results are saved to the current folder in {}_Con_Results.csv.".format(folder_name[:3]))
print('-'*20)
csv_file = "{}_Con_Results.csv".format(folder_name[:3])
csv_columns = ['id', 'file_name', 'mean', 'std', 'otsu', 'channel']
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(results)
except:
    print("Cannot saving to csv file")


# In[ ]:




