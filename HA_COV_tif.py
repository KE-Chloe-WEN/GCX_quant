#!/usr/bin/env python
# coding: utf-8

# In[57]:


import os
import numpy as np
from czifile import CziFile
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import LineString
import tifffile as tif
import csv


# In[58]:


def get_tif_files(ext = 'tif', con = '.'):
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
    
def find_nonzeropoint(multipoint):
    """
    This function returns all points whose x or y coordinates are not zero.
    """
    juncs = list(multipoint.geoms)
    leng = len(juncs)

    for i in range(1, len(juncs)+1):
        leng = len(juncs) - i
        if juncs[leng].y > 0:
            return (juncs[leng])
            break


# In[59]:


# os.chdir('/Users/kewen/Library/CloudStorage/OneDrive-NortheasternUniversity/Coverage Images for HBMECs and HPMECs/HBMEC/WGA/9.18.2023 HBMEC P6 6hr ST F 30dynescm2 WGA')
NC_files = get_tif_files(con = 'NC')
# Control image file.
CON_files = get_tif_files(con = 'CON') 
# Sample image file.
im_files = get_tif_files(con = 'hr') 
# Define the channel index will be extracted.
channel_index = 0
# Save folder name of the current path.
folder_name = os.getcwd().split('/')[-1] 
results = [] 

print("Finding threshold ... ")
NC_arrays = tif.imread(NC_files)
CON_arrays = tif.imread(CON_files)
NC = NC_arrays[channel_index]
CON = CON_arrays[channel_index]

# Find the max bit size of the control. 
bits = np.floor(np.log2(np.max(CON)))
# bin size 500, should decrease or increase based on needs of resolution. 
bins = np.linspace(0,2**bits,500)
# Reduce number of elements in the x axis, to match the number of y value. 
xs = ( bins[:-1] + bins[1:] )/2

# Plot histogram of frequency vs. intensity for negative control. x axis - intensity, y axis - frequency. 
y0,_ = np.histogram(NC.reshape(-1,1), bins = bins) # reshape (x, y, z), -1 is a sign for the entire dimension
# Plot histogram of frequency vs. intensity for control. x axis - intensity, y axis - frequency.
y1,_ = np.histogram(CON.reshape(-1,1), bins = bins)

plt.plot(xs,y0,'r',label='Negative Control')
plt.plot(xs,y1,'b',label='Control')
plt.legend()

# Set x, y limits.
plt.xlim([0,30000])
plt.ylim([0,100000])

# Find threshold = intersection of negative control and control, intensity.
line1 = LineString(np.column_stack((xs, y0)))
line2 = LineString(np.column_stack((xs, y1)))
inters = find_nonzeropoint(line1.intersection(line2))
plt.plot(inters.x, inters.y, 'o', color="green")
threshold = inters.x

plt.savefig('Threshold.png', bbox_inches='tight')
# plt.show()

# Use threshold to find percent coverage on image samples. 
for im_file in im_files:
    print(" - Processing image {} ...\n".format(im_file), end="")
    
    im_arrays = tif.imread(im_file)
    im = im_arrays[channel_index]
    
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
    
    results.append({"file_name" : im_file, "cov" : covp, "threshold": threshold}) 
print(" Finished.")

print("\n Analysis finished, results are saved to the current folder in {}_Single_Con_Results.csv.".format(folder_name[:4]))
print('-'*20)
csv_file = "{}_Single_Con_Results.csv".format(folder_name[:4])
csv_columns = ['file_name', 'cov', 'threshold']

try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(results)
except:
    print("Cannot saving to csv file")


# In[ ]:




