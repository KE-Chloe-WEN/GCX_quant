#!/usr/bin/env python
# coding: utf-8

# In[24]:


import os
import numpy as np
from czifile import CziFile
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import LineString
import csv


# In[25]:


def get_czi_files(ext = 'czi', con = '.'):
    """ This function returns all czi files under current folder as a list.
    input extension: extension of file needs to be filtered
    input contains: file name must contain string.
    """
    results = []
    files = os.scandir()
    for file in files:
        if file.is_file and file.name.endswith(ext) and con in file.name:
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

def find_nonzero(multipoint):
    """
    This function returns all points whose x or y coordinates are not zero.
    """
    if type(multipoint) is shapely.geometry.multipoint.MultiPoint:
        for p in multipoint.geoms:
            if p.x > 0 and p.y > 0:
                return p
    else:
        return multipoint


# In[26]:


# os.chdir('/Users/kewen/OneDrive - Northeastern University/HA/4.8.2022 HBMEC P7 6hr F+S 12 dynescm2')
# Negative control file under current path.
NC_files = get_czi_files(con = 'NC')
# Sample image file.
im_files = get_czi_files(con = 'hr') 
# Define the channel index will be extracted.
channel_index = 0
# Save folder name of the current path.
folder_name = os.getcwd().split('/')[-1] 
results = [] 

NC_file = NC_files[0]

print("Finding threshold ... \n")

# Save 5d/6d array from image.
with CziFile(NC_file) as czi:
    NC_arrays = czi.asarray() 
    
# Drop empty dimensions in the array.
NC_arrays = drop_empty_dim(NC_arrays)
# z-stack slices for xy image in green channel.
NC = NC_arrays[channel_index] 
# Find maximum intensity in the z-projection. Final NC should be a 2d array. 
NC = np.max(NC, axis=0)

# Use threshold to find percent coverage on image samples. 
for im_file in im_files:
    print(" - Processing image {} ...\n".format(im_file), end="")
    
    # Save 5d/6d array from image.
    with CziFile(im_file) as czi:
        im_arrays = czi.asarray() 
    
    im_arrays = drop_empty_dim(im_arrays)
    im = im_arrays[channel_index]
    im = np.max(im, axis=0)
    
    # Find the max bit size of the control. 
    bits = np.floor(np.log2(np.max(im)))
    # bin size 500, should decrease or increase based on needs of resolution. 
    bins = np.linspace(0,2**bits,500)
    # Reduce number of elements in the x axis, to match the number of y value. 
    xs = ( bins[:-1] + bins[1:] )/2

    # Plot histogram of frequency vs. intensity for negative control. x axis - intensity, y axis - frequency. 
    y0,_ = np.histogram(NC.reshape(-1,1), bins = bins) # reshape (x, y, z), -1 is a sign for the entire dimension
    # Plot histogram of frequency vs. intensity for control. x axis - intensity, y axis - frequency.
    y1,_ = np.histogram(im.reshape(-1,1), bins = bins)

    plt.plot(xs,y0,'r')
    plt.plot(xs,y1,'b')

    # Set x, y limits.
    plt.xlim([0,10000])
    plt.ylim([0,100000])
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    # Find threshold = intersection of negative control and control, intensity.
    line1 = LineString(np.column_stack((xs, y0)))
    line2 = LineString(np.column_stack((xs, y1)))
    inters = find_nonzero(line1.intersection(line2))
    plt.plot(inters.x, inters.y, 'o', color="green")
    threshold = inters.x

    plt.savefig('Threshold{0}.png'.format(im_file), bbox_inches='tight')
    # plt.show()
    plt.close()
    
    # Find row and column numbers in czi image. 
    row, col = im.shape
    
    # Total number of pixels.
    Tot = row*col

    # Thresholding with the Tarbell method.
    im = im * (im > threshold)
    
    # Find total pixels with more than 0 intensity, area covered by ROI.
    length = np.arange(0, col, 1, dtype=int)
    cov = np.sum(im[:, length] !=0, axis=0)
    covt = np.sum(cov)
    covp = covt/Tot*100
                
    results.append({"file_name" : im_file, "coverage %" : covp, "threshold": threshold}) 
print(" Finished.")

print("\n Analysis finished, results are saved to the current folder in {}_Con_Results.csv.".format(folder_name[:3]))
print('-'*20)
csv_file = "{}_Per_Con_Results.csv".format(folder_name[:4])
csv_columns = ['file_name', 'coverage %', 'threshold']
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(results)
except:
    print("Cannot saving to csv file")

