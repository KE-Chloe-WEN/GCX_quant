import os
import numpy as np
from czifile import CziFile
import matplotlib.pyplot as plt
from skimage import filters
from scipy.ndimage import gaussian_filter
import shapely
from shapely.geometry import LineString
import csv

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
    
def cutout_blank(arr):
    return arr [:, arr.sum(axis = 0) !=0]


# update the next line file path to where you uploaded your files.
#os.chdir('file name')
im_files = get_czi_files() 
channel_index = 0
# Number of lines
k = 100 
# Standard deviation for Gaussian kernel
simga_gauss = 1
folder_name = os.getcwd().split('/')[-1] 
results = []

for im_file in im_files:
    print(" - Processing image {} ...\n".format(im_file))
    case = 0
    bthick = []
    athick = []
    sthick = []
    # Save 5d/6d array from image.
    with CziFile(im_file) as czi:
        im_arrays = czi.asarray() 
        
    im_arrays = drop_empty_dim(im_arrays)
    im = im_arrays[channel_index]
    im = im[:,:,im.shape[2]//2]
    # plt.imsave('{}-{}.png'.format(im_file,"Before SA"), im, cmap='gray') # Delete when code is complete 
    # Apply gaussian blur to image. 
    im_clean = gaussian_filter(im, simga_gauss) 
    otsu_threshold = filters.threshold_otsu(im_clean)
    im_clean = im_clean * (im_clean > otsu_threshold)
    #print(otsu_threshold)
    
    # Cut out blank space after filtering.
    im_clean = cutout_blank(im_clean) 
    plt.imsave('{}-{}.png'.format(im_file,"SA"), im_clean, cmap='gray') # Delete when code is complete 
    
    # Mimic drawing k number of random lines on the image to extract fluorescence thickness.
    rand_index = np.random.randint(im_clean.shape[1], size = k)
    
    for i in range(0,k,1): # Use k as defined eariler for # of lines.
        rand_arr = im_clean[:, rand_index[i]]
        #plt.plot(rand_arr)
        x = np.arange(0,rand_arr.size)
        y = np.ones(rand_arr.size)
        #y = np.full((rand_arr.size), 1, dtype=int)
        #plt.plot(y)
        #plt.savefig('{}-{}.png'.format(im_file,i))

        # Find intersections of the intensity profile and intensity = 1.
        first_line = LineString(np.column_stack((x, y)))
        second_line = LineString(np.column_stack((x, rand_arr)))
        inters = first_line.intersection(second_line)
        # print(inters)
        
        if type(inters) is shapely.geometry.multipoint.MultiPoint:
            
            leninters = len(inters.geoms)
            p = []
            for point_ in inters.geoms: # Loop list/array directly without using index.
                p.append(point_.x)
            p.sort()
            
            # --- Equal to the following:
            
            #leninters = len(inters.geoms)
            #p = [point_.x for point_ in inters.geoms].sort() # This is called list comprehension in python.
            
            if leninters > 2:
                #plt.plot(rand_arr) # Delete when code is complete 
                #plt.plot(y) # Delete when code is complete 
                #plt.savefig('{}-{}.png'.format(im_file,i)) # Delete when code is complete 
                #plt.close() # Delete when code is complete 
                
                if leninters == 3:
                    #plt.plot(rand_arr) # Delete when code is complete 
                    #plt.plot(y) # Delete when code is complete 
                    #plt.savefig('{}-{}.png'.format(im_file,i)) # Delete when code is complete 
                    #plt.close()
                    if rand_arr[0]>0:
                        splitpoint = int(p[0]) 
                    else:
                        splitpoint = int(p[-1]) 

                if leninters == 4: 
                    #p = []
                    #for j in range(0,leninters):
                    #    p.append (inters.geoms[j].x)
                    #p.sort()
                    #case = case + 1
                    splitpoint = int(p[1]) # Split array at intersections.

                if leninters > 4: # If there is more than 4 intersections, figure is considerred invalid.
                    break

                rand_split = np.array_split(rand_arr, [splitpoint+1])
                basal = rand_split[0]
                apical = rand_split[-1]
                thickb = np.sum(basal[:] !=0)
                thicka = np.sum(apical[:] !=0)
                bthick.append(thickb)
                athick.append(thicka)
                case = case + 1
            
            else:
                thicks = np.sum(rand_arr[:] !=0)
                #print(thicks)
                sthick.append(thicks)
            
    lenb = len(bthick)
    lena = len(athick)
    lens = len(sthick)
    #print (bthick, lenb)
    #print (athick, lena)
    
    if lenb == 0:
        AVEbthick = 0
    else:
        AVEbthick = ((sum(bthick)/lenb)-1)*0.38 #µm
        
    if lena == 0:
        AVEathick = 0
    else:
        AVEathick = ((sum(athick)/lena)-1)*0.38 #µm
        
    if lens == 0:
        AVEsthick = 0
    else:
        AVEsthick = ((sum(sthick)/lens)-1)*0.38 #µm
        
    portion = case / k * 100 # percentage of cases that the expression shows apical and basal portions

    # Dictionary, link index with a string...
    results.append({"file_name" : im_file,
                    "Average Basal Thickness(µm)" : AVEbthick,
                    "Average Apical Thickness(µm)" : AVEathick,
                    "Average Single Thickness(µm)" : AVEsthick,
                    "Chances of Bilayer Expression(%)" : portion,
                    "Threshold": otsu_threshold,
                    "Gaussian": simga_gauss}) 
    
print(" Finished.")

csv_file = "{}_Con_Results.csv".format(folder_name[:3])
csv_columns = ['file_name', 'Average Basal Thickness(µm)', 'Average Apical Thickness(µm)', 'Average Single Thickness(µm)', 'Chances of Bilayer Expression(%)', 'Threshold', 'Gaussian']

try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(results)
    
    # Print only if the csv file is created.
    #print("\n Analysis finished, results are saved to {}_Con_Results.csv under {}.".format(folder_name[:3], os.getcwd()))
    #print('-'*20)    
    print("")
    print("-"*20)
    print(f"Results are saved to {os.getcwd()}/{csv_file}. ")
    print("-"*20)
    print("")
except:
    print("Cannot saving to csv file")