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
    """ This function is used to cut out any blank areas in the first dimension of the image 
    To avoid drawing lines in an empty space.
    """
    return arr [:, arr.sum(axis = 0) !=0]


# Update the next line file path to where the image files are kept.
# os.chdir('Folder Name') # Substitute Folder Name with the pathname.
im_files = get_czi_files() 

# Open the first channel for the green channel - Glycocalyx.
channel_index = 0

# Define number of lines.
k = 100 

# Define standard deviation for Gaussian kernel.
simga_gauss = 1
folder_name = os.getcwd().split('/')[-1] 
results = []

# Open image file under the file path. 
for im_file in im_files:
    print(" - Processing image {} ...\n".format(im_file))
    
    # Define case to count the occurance of multilayer expression in the sample set. 
    case = 0
    
    # Define lists to contain thickness data for basal, apical, and single layer expression. 
    bthick = []
    athick = []
    sthick = []
    
    # Save 5d/6d array from image.
    with CziFile(im_file) as czi:
        im_arrays = czi.asarray() 
        
    im_arrays = drop_empty_dim(im_arrays)
    im = im_arrays[channel_index]
    
    # Obtain orthogonal view of the Z-stack image. XZ view at the midpoint in the Y axis. 
    im = im[:,:,im.shape[2]//2]
    
    # plt.imsave('{}-{}.png'.format(im_file,"Before SA"), im, cmap='gray') # This line is used to print original image.
    
    # Apply gaussian blur to the image and find otsu threshold. 
    im_clean = gaussian_filter(im, simga_gauss) 
    otsu_threshold = filters.threshold_otsu(im_clean)
    
    # Apply threshold to zero all pixels whose intensity value is less than the threshold.
    im_clean = im_clean * (im_clean > otsu_threshold)
    
    # Cut out blank space after filtering.
    im_clean = cutout_blank(im_clean) 
    
    # plt.imsave('{}-{}.png'.format(im_file,"SA"), im_clean, cmap='gray') # This line is used to print image after filtering.
    
    # Mimic drawing k number of random lines on the image to extract fluorescence thickness.
    rand_index = np.random.randint(im_clean.shape[1], size = k)
    
    for i in range(0,k,1): # Use k as defined eariler for # of lines.
        rand_arr = im_clean[:, rand_index[i]]
        x = np.arange(0,rand_arr.size)
        y = np.ones(rand_arr.size)
        
        # Find intersections of the intensity profile and intensity = 1 to find the layers of expression in the image.
        first_line = LineString(np.column_stack((x, y)))
        second_line = LineString(np.column_stack((x, rand_arr)))
        inters = first_line.intersection(second_line)
        
        # Find the coordinates of the intersections. 
        if type(inters) is shapely.geometry.multipoint.MultiPoint:
            leninters = len(inters.geoms)
            p = []
            
            for point_ in inters.geoms: 
                p.append(point_.x)
            p.sort()
            
            # --- Equal to the following:
            
            #leninters = len(inters.geoms)
            #p = [point_.x for point_ in inters.geoms].sort() # This is called list comprehension in python.
            
            # If there are more than two intersections, glycocalyx is expressed more than a singel layer.
            if leninters > 2:
                # Find split points between intersections to define basal and apical layers.
                if leninters == 3:
                    if rand_arr[0]>0:
                        splitpoint = int(p[0]) 
                    else:
                        splitpoint = int(p[-1]) 

                if leninters == 4:
                    splitpoint = int(p[1])
                    
                # If there is more than 4 intersections, the expression is considered as noisy and image is considerred invalid.
                if leninters > 4: 
                    break

                # Split arrays at the split point to separate basal and apical expressions.
                rand_split = np.array_split(rand_arr, [splitpoint+1])
                basal = rand_split[0]
                apical = rand_split[-1]
                
                # Sum non-zero pixels along the line for thickness calculation for both the basal and the apical layers.
                thickb = np.sum(basal[:] !=0)
                thicka = np.sum(apical[:] !=0)
                bthick.append(thickb)
                athick.append(thicka)
                
                # Count the case of occurance for multi-layer expression.
                case = case + 1
            
            else:
                # If only two intersections are found, the image only has single-layer expression. 
                # Sum non-zero pixels along the line for thickness calculation.
                thicks = np.sum(rand_arr[:] !=0)
                #print(thicks)
                sthick.append(thicks)
    
    # If the list for thickness counting is empty, no thickness value is found. Record value as zero. 
    lenb = len(bthick)
    lena = len(athick)
    lens = len(sthick)
    
    if lenb == 0:
        AVEbthick = 0
    else:
        # Convert the count of non-zero value to thickness value by multiplying the length of the interval between z-stacks. 
        AVEbthick = ((sum(bthick)/lenb)-1)*0.38 #µm
        
    if lena == 0:
        AVEathick = 0
    else:
        # Convert the count of non-zero value to thickness value by multiplying the length of the interval between z-stacks. 
        AVEathick = ((sum(athick)/lena)-1)*0.38 #µm
        
    if lens == 0:
        AVEsthick = 0
    else:
        # Convert the count of non-zero value to thickness value by multiplying the length of the interval between z-stacks. 
        AVEsthick = ((sum(sthick)/lens)-1)*0.38 #µm
    
    # Percentage of cases that the expression shows apical and basal portions.
    portion = case / k * 100 

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
   
    print("")
    print("-"*20)
    print(f"Results are saved to {os.getcwd()}/{csv_file}. ")
    print("-"*20)
    print("")
except:
    print("Cannot saving to csv file")
