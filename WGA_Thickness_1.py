#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

def drop_empty_dim(arr: np.array) -> np.array:
    '''
    This function is used for dropping empty dimension (shape = 1) of the input array.
    E.g., input array shape = [1, 1, 3, 4, 5] ==> [3, 4, 5] as output array shape
    
    '''
    
    dims_to_sq = np.where(np.array(arr.shape) == 1)[0]
    for dim_ in dims_to_sq[::-1]:
        arr = np.squeeze(arr, axis= dim_)
        
    return arr

def cutout_blank(arr):
    return arr [:, arr.sum(axis = 0) !=0]

# czi files under current path.
czi_files = get_czi_files() 
# Number of lines
k = 50 
# Standard deviation for Gaussian kernel
simga_gauss = 1
# Define the channel index will be extracted.
channel_index = 0 
# Save folder name
folder_name = os.getcwd().split('/')[-1]
n = 1
# Create result list
results = [] 

print('-'*20)
print(" Params used:\n")
print(" - Number of random lines used: {}".format(k))
print(" - Gaussian blur: {}".format(simga_gauss))
# print(" - All channel analysis mode")
print(" - Channel of czi image: {}\n".format(channel_index))
print('-'*20)
print(" Image processing in progress ...\n")

for czi_file in czi_files:
    print(" - Processing image {} ...\n".format(czi_file), end="")
    
    # Save 5d/6d array from image.
    with CziFile(czi_file) as czi:
        image_arrays = czi.asarray() 
    
    # Drop dimensions whose size is 1.
    image_arrays = drop_empty_dim(image_arrays)
    
    # Save defined channel to im.
    im = image_arrays[channel_index]

    # Flip all orthogonal view images to x-z.
    row, col = im.shape
    if row > col:
        im = np.transpose(im)

    # Apply gaussian blur to image. 
    im_clean = gaussian_filter(im, simga_gauss) 
    otsu_threshold = filters.threshold_otsu(im_clean)
    im_clean = im_clean * (im_clean > otsu_threshold)

    # Cut out blank space after filtering.
    # im_clean = cutout_blank(im_clean) 

    # Mimic drawing k number of random lines on the image to extract fluorescence thickness.
    rand_index = np.random.randint(im_clean.shape[1], size = k)

    # Use the following script if want to save modified image with random lines shown.
    im_plot = im_clean.copy()
    #im_plot[:, rand_index] = np.max(im_plot) + 500
    #im_clean[:, rand_index] = np.max(im_clean) 
    plt.imsave('{}-{}.png'.format(czi_file.split('.')[0], simga_gauss), im_plot, cmap='gray')

    # Average thickness in microns.
    rand_thick = np.sum(im_clean[:, rand_index] !=0, axis=0) * 9.17 / 130 
    thick_mean = np.average(rand_thick)
    thick_std = np.std(rand_thick)

    # Dictionary, link index with a string...
    results.append({"file_name" : czi_file, "mean" : thick_mean, "std" : thick_std, "otsu" : otsu_threshold, "gaussian": simga_gauss}) 
print(" Finished.")

print("\n Analysis finished, results are saved to the current folder in {}_Con_Results.csv.".format(folder_name[:3]))
print('-'*20)
csv_file = "{}_Con_Results.csv".format(folder_name[:3])
csv_columns = ['file_name', 'mean', 'std', 'otsu','gaussian']
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(results)
except:
    print("Cannot saving to csv file")

