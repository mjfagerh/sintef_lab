import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import pandas as pd
from scipy import ndimage
from skimage.draw import line
from scipy.signal import find_peaks
from numba import jit
import math



def get_mask():
    return np.load("../raw_files/mask.npz")['arr_0']
def invert(image):
    return np.where(image==1,0,1)
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]
def qplot(image):
    plt.figure()
    plt.title(namestr(image, globals()))
    plt.imshow(image)
# the labels from mask is applyed on Asub2


def get_labeled_mask(mask):
    mask_label, num1 = label(mask, connectivity=2, return_num=True)
    return mask_label, num1

def get_Asub_label(Asub):
    Asub_label, num2 = label(Asub, connectivity=2, return_num=True)
    return Asub_label, num2

def label_with_mask(labeled_mask, Asub):
    re_Asub = np.where(Asub != 0, labeled_mask, Asub)
    return re_Asub

def get_max_label(label_field):
    return np.max(label_field)


def get_lable_values(label_field):
    a = np.unique(label_field)
    return a[a!=0]

def get_num_of_lables(label_field):
    return np.count_nonzero(get_lable_values(label_field))



def isolate_obj(labeled_mask,labeled_Asub, pix_val):
    return np.where(labeled_mask == pix_val, labeled_Asub, 0)

## gives areas form regionprops
def Area_difference(mask_props, Asub_props):
    objects_found = len(mask_props)
    mask_Areas = np.zeros(objects_found)
    Asub_Areas = np.zeros(objects_found)
    for i in range(objects_found):
        mask_Areas[i] = mask_props[i].area
        Asub_Areas[i] = Asub_props[i].area
    total_relative_area_dif = (np.sum(mask_Areas)-np.sum(Asub_Areas))/np.sum(mask_Areas)
    local_relative_area_dif = (mask_Areas-Asub_Areas)/mask_Areas
    return total_relative_area_dif, local_relative_area_dif

# tell us the number of objects difference (small islands will be counted)

def num_islands_per_obj(labeled_mask, labeled_Asub, num_mask_obj):
    count = np.zeros(num_mask_obj)
    for pix_val in range(1,num_mask_obj+1):
        obj = isolate_obj(labeled_mask, labeled_Asub, pix_val)
        num_labels = get_num_of_lables(obj)
        count[pix_val-1] = num_labels-1
    return count

# counts number of object that should be in asub by using mask. then filters away the smaller areas
# we are lef with what we would assume to be the main tube

def filter_away_islands(labeled_mask, labeled_Asub, num_mask_obj):
    mask_labels = get_lable_values(labeled_mask)
    index = np.zeros(num_mask_obj)
    for i, pix_val in enumerate(mask_labels):
        A = 0
        obj = isolate_obj(labeled_mask,labeled_Asub, pix_val)
        obj_labels = get_lable_values(obj)
        for val in obj_labels:
            a = np.count_nonzero(obj == val)
            if A < a:
                A = a
                index[i] = val
    return np.where(np.isin(labeled_Asub,index), labeled_mask, 0)

def get_CM(matrix):
    return ndimage.center_of_mass(matrix)

#not in use so far
def elips_axis_differencte_2D(mask_props, Asub_props):
    objects_found = len(mask_props)
    objects_delta_minor = np.zeros(objects_found)
    objects_delta_major = np.zeros(objects_found)
    obj_dif = objects_found-len(Asub_props)
    for i in range(obj_dif):
        mask_props.sort(key=lambda x: x.area)
        mask_props.pop(i)
    for i in range(objects_found-obj_dif):
        objects_delta_minor[i] = mask_props[i].axis_minor_length-Asub_props[i].axis_minor_length
        objects_delta_major[i] = mask_props[i].axis_major_length-Asub_props[i].axis_major_length

    return objects_delta_major, objects_delta_minor

@jit
def dist_points(x0,y0,x1,y1):
    dist = (np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2))
    if isinstance(dist, np.floating):
        return dist
    return min(dist)

# sort points to preserve neighbourhoods based on distance
@jit
def sort_coordinates(coord):
    coord = list(zip(coord[0],coord[1]))

    sorted_list = [coord.pop(0)]
    while len(coord) != 0:
        x0, y0 = sorted_list[-1]
        D = 8e8 # just a Big number
        i_to_pop=0
        for i, (x1,y1) in enumerate(coord):
            d = dist_points(x0,y0,x1,y1)
            if d<D:
                D = d
                i_to_pop = i
        sorted_list.append(coord.pop(i_to_pop))
    return np.array(sorted_list)

def map_dif_dist(labeled_mask, labeled_Asub, mask_props):
    rand_mask = labeled_mask.copy()
    rand_Asub = labeled_Asub.copy()


    rand_mask[ndimage.binary_erosion(rand_mask.tolist())] = 0
    rand_Asub[ndimage.binary_erosion(rand_Asub.tolist())] = 0
    total_error = []
    for t, mprops in enumerate(mask_props):
        cmy, cmx = mprops.centroid
        cmy, cmx = int(cmy), int(cmx)
        mlabel_value = mprops.label
        coordinates = sort_coordinates(np.where(rand_mask == mlabel_value))
        step_lenght = 1
        num_valid_coord = len(coordinates)
        r = np.zeros(math.floor(num_valid_coord/step_lenght))
        for index, (i, j) in enumerate(coordinates):
            if index%step_lenght != 0:
                continue
            line_value = 10 #arbitrary, but high enough to not conflict wiht other values in array
            rr, cc = line(i,j,cmy,cmx)
            rand_Asub[rr,cc] += line_value

            coord_error = np.where(rand_Asub == mlabel_value+line_value)
            if not np.any(coord_error):
                r[index] = r[index-1]
                continue
            r[index] = dist_points(i, j, coord_error[0], coord_error[1])


            rand_Asub[rand_Asub >= line_value] = 0

            rand_Asub[coord_error[0], coord_error[1]] = mlabel_value
        total_error.append(r)
    return total_error

def get_signal_stats(stats):
    N = len(stats)
    if N==0:
        return np.array([0]),np.array([0]),np.array([0])
    max_error = np.zeros(N)
    num_error_peaks = np.zeros(N)
    avg_error = np.zeros(N)
    for j in range(N):
        data = stats[j]
        peaksx, _ = find_peaks(data)
        max_error[j] = np.max(data)
        num_error_peaks[j] = len(peaksx)
        if len(peaksx) == 0:
            avg_error[j] = np.max(data)
        else:
            avg_error[j] = np.average(data[peaksx])

    return max_error, num_error_peaks, avg_error

def get_radial_dist(labeled_mask,mask_props):
    imagex, imagey = labeled_mask.shape
    centerx = int(imagex/2)
    centery = int(imagey/2)
    radial_dist = np.zeros(len(mask_props))
    for i,props in enumerate(mask_props):
        cmy, cmx = props.centroid
        cmy,cmx = int(cmy), int(cmx)
        radial_dist[i] = dist_points(cmx,cmy,centerx,centery)
    return radial_dist


def creat_stats(name, mask, sub,labeled_mask1,num_objects1,mask_props1):
    x, y, z = mask.shape
    start = 0
    end=z
    skip = 1
    total_relative_area_dif = np.zeros(z,dtype=object)
    local_relative_area_dif = np.zeros(z,dtype=object)
    num_islands = np.zeros(z,dtype=object)
    max_error = np.zeros(z,dtype=object)
    num_error_peaks = np.zeros(z,dtype=object)
    avg_error = np.zeros(z,dtype=object)
    CM_shift = np.zeros(z,dtype=object)
    relative_volum_dif = np.zeros(z,dtype=object)
    analysed_slices_index = np.zeros(z, dtype=object)
    radial_dist = np.zeros(z, dtype=object)
    indexes = []

    print(name)
    for i in range(start,end,skip):

        if i%50 ==0:
            print(i)
        try:
        #if True:

            Asub_slice = sub[:, :, i]

            labeled_mask = labeled_mask1[i]
            num_objects = num_objects1[i]
            if np.max(labeled_mask) == 0:
                print(f"skiped slice{i}")
                continue
            labeled_Asub = label_with_mask(labeled_mask,Asub_slice)
            labeled_Asub_no_mask,num_Asub_objects = get_Asub_label(Asub_slice)
            labeled_asub_main = filter_away_islands(labeled_mask, labeled_Asub_no_mask, num_objects)
            mask_props = mask_props1[i]
            Asub_props = regionprops(labeled_Asub, cache=True)
            if len(mask_props) != len(Asub_props):
                print(f"un-even number of objects for slice {i}")
                continue
            Asub_main_area_props = regionprops(labeled_asub_main, cache=True)

            radial_dist[i-start] = get_radial_dist(labeled_mask,mask_props)
            total_relative_area_dif_slice, local_relative_area_dif_slice = Area_difference(mask_props, Asub_main_area_props)
            total_relative_area_dif[i - start] = total_relative_area_dif_slice
            local_relative_area_dif[i - start] = local_relative_area_dif_slice
            num_islands[i-start] = num_islands_per_obj(labeled_mask, labeled_Asub_no_mask, num_objects)
            max_error[i-start], num_error_peaks[i-start], avg_error[i-start] = get_signal_stats(map_dif_dist(labeled_mask, labeled_asub_main, mask_props))
            indexes.append(i)
        except Exception as e:
            print(f"exception at slice {i}")
            print(e)
            continue
    analysed_slices_index[0] = str(indexes)
    mask_volum = np.count_nonzero(mask)
    sub_volum = np.count_nonzero(sub)
    relative_volum_dif[0] = str((mask_volum-sub_volum)/mask_volum)
    CM_shift[0] = np.array(get_CM(sub))-np.array(get_CM(mask))
    dict_stat = {
        "Total relativ area difference": total_relative_area_dif,
        "Local area difference": local_relative_area_dif,
        "Number of islands": num_islands,
        "Relative Volum difference": relative_volum_dif,
        "CM shift Volume": CM_shift,
        "Max error": max_error,
        "Num error peaks": num_error_peaks,
        "Avg error": avg_error,
        "Analysed slices index": analysed_slices_index,
        "Radial distance": radial_dist
    }
    df = pd.DataFrame(data=dict_stat)
    df.to_csv(f"{name}.csv")

def mask_analasys():
    return 0
for i in (["50_proj","55_proj","60_proj","65_proj","70_proj","75_proj","80_proj","85_proj","90_proj","95_proj","100_proj","150_proj","300_proj","600_proj","1200_proj","3142_proj","mask"]):
    print(i)

mask = invert(get_mask())
x, y, z = mask.shape
start = 0
end = z
skip = 1
labeled_mask = np.zeros(z,dtype=object)
num_objects = np.zeros(z,dtype=object)
mask_props = np.zeros(z,dtype=object)
for i in range(start, end, skip):
    mask_slice = mask[:, :, i]
    labeled_mask[i], num_objects[i] = get_labeled_mask(mask_slice)
    mask_props[i] = regionprops(labeled_mask[i], cache=True)



for name in reversed(["50_proj","55_proj","60_proj","65_proj","70_proj","75_proj","80_proj","85_proj","90_proj","95_proj","100_proj","150_proj","300_proj","600_proj","1200_proj","3142_proj","mask"]):
    sub = np.load(f"{name}.npz")['arr_0']
    if name == "mask":
        sub = invert(sub)


    creat_stats(f"stats_{name}", mask,sub,labeled_mask,num_objects,mask_props)
















