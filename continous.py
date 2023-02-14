import os
import shutil
import numpy as np
import imageio.v2 as imageio
import tifffile as tf
import matplotlib.pyplot as plt
from Create_filesystem_for_projections import get_fil_type, get_fil_num, move_xtekct_fil,adjust_xtekct_fil
from mask_image import qplot
main_dest = "E:\Mats_08\CT-scanns\groundTrouth_TwoPoreSystem"

source_dir = f"{main_dest}\\raw_data"
base_name = "KS20221003_MJHF_groundtruth_"
total_number_of_images_used = 0





def give_even_dist_angles_deg(N):
    """

    :param int N: number of images that is to be distributed
    :return: and array of the average angular postions of the images
    """
    wp = np.linspace(0, 360, N, endpoint=False, dtype=int)
    return wp


def give_sector_end_points(mid_angle, sec_angle):
    return np.array([mid_angle-sec_angle/2, mid_angle+sec_angle/2])


def angle_to_proj(ang, N_images = 3142):
    val = round((ang)/(360/N_images))%N_images
    return val


def matrix_to_tiff(mat, num, end_dest):
    """

    :param mat: image matrix
    :param num: image number
    :param end_dest: the new created director
    :return:
    """
    path = os.path.join(end_dest, base_name+str(num).zfill(4)+".tif")
    tf.imwrite(path, mat)


def get_image_array(image):
    return tf.imread(image)


def create_dir(end_dest):
    if os.path.exists(end_dest):
        shutil.rmtree(end_dest)
    os.mkdir(end_dest)


def get_average_proj(sos, eos):
    global total_number_of_images_used
    avg_mat = np.zeros((2000, 2000)) ## found the dimentions of the tif images
    files = os.listdir(source_dir)
    files = [i for i in files if get_fil_type(i) =='tif']
    if eos >= sos:
        files = [i for i in files if (get_fil_num(i)>sos and get_fil_num(i)<eos) or i == sos]
    else:
        files = [i for i in files if get_fil_num(i) >= sos or get_fil_num(i) < eos]

    for fil in files:
        total_number_of_images_used += 1
        avg_mat += get_image_array(os.path.join(source_dir, fil))

    return np.uint16((avg_mat/(len(files))))


def sectors_from_full_image(num_sectors):
    for i in num_sectors:
        N_images = 3142
        cake_pieces = i
        images_per_piece = int(N_images / cake_pieces)
        images_remaining = N_images % cake_pieces
        end_dest = os.path.join(main_dest, f"continous_{cake_pieces}_sectors")
        create_dir(end_dest)

        for i in range(cake_pieces):
            start = images_per_piece*i+1
            end = start+images_per_piece
            matrix_to_tiff(get_average_proj(start,end), i+1, end_dest)

        if images_remaining != 0:
            start = end
            end = start+images_remaining
            matrix_to_tiff(get_average_proj(start, end), cake_pieces,end_dest)

        adjust_xtekct_fil(end_dest,move_xtekct_fil(end_dest),cake_pieces)


def discrete_proj_with_sectors(proj_angles,sector_ang):
    give_sector_warning(proj_angles, sector_ang)
    end_dest = os.path.join(main_dest, f"wp")
    create_dir(end_dest)
    for i in range(len(proj_angles)):
        start, end = give_sector_end_points(proj_angles[i], sector_ang)
        start = angle_to_proj(start)
        end = angle_to_proj(end)
        matrix_to_tiff(get_average_proj(start, end), i + 1, end_dest)
    adjust_xtekct_fil(end_dest, move_xtekct_fil(end_dest), len(proj_angles))
    print(f"used {total_number_of_images_used} images in total ")
    new_dir_name = os.path.join(main_dest,f"{len(proj_angles)}_projection angles_with_sector_angle_{sector_ang}__{total_number_of_images_used}_images_used")
    if os.path.exists(new_dir_name):
        shutil.rmtree(new_dir_name)
    os.rename(end_dest, new_dir_name)

def images_used(proj_ang, sec_ang):
    start, end = give_sector_end_points(proj_angles[0], sector_ang)
    start = angle_to_proj(start)
    end = angle_to_proj(end)
    val = 0
    if end>start:
        val = end-start
    else:
        val = 3142-(start-end)
    val*=len(proj_ang)
    print(f"total number of images used is {total_number_of_images_used}")
    return val


def give_sector_warning(proj_ang, sector_ang):
    for i in range(len(proj_ang)):
        start1, end1 = give_sector_end_points(proj_ang[i], sector_ang)
        if start1>end1:
            print("Warning! one of you sectors averages pictures above and below 0.")
        for j in range(i, len(proj_ang)):
            start2, end2 = give_sector_end_points(proj_ang[j],sector_ang)
            if i!=j and end1>start2:
                print("Warning! The sectors overlapp")


def plot_covered_area(proj_angles, sector_ang):
    proj_angles = np.radians(proj_angles)
    sector_ang = np.radians(sector_ang)
    all_ang = np.linspace(0, np.pi * 2, 3142)
    plt.figure(figsize=(5, 5))
    plt.plot(np.cos(all_ang), np.sin(all_ang))
    for mid in proj_angles:
        plt.plot([0,np.cos(mid)],[0,np.sin(mid)])
        sos, eos = give_sector_end_points(mid, sector_ang)
        plt.plot([0, np.cos(sos)], [0, np.sin(sos)], color = 'Black')
        plt.plot([0, np.cos(eos)], [0, np.sin(eos)], color = 'Black')
    plt.show()



# integrate over the images in sector batches
if __name__ == "__main__":
    number_of_sectors = 150
    #getting basic informations
    files = os.listdir(source_dir)
    image_files = [i for i in files if get_fil_type(i) == 'tif']
    number_of_images = len(image_files)

    #fixing file system, and moving xteckfile
    end_dest = os.path.join(main_dest, f"integrating_images__{number_of_sectors}_sectors")
    create_dir(end_dest)
    moved_xtekct_name = move_xtekct_fil(end_dest)
    adjust_xtekct_fil(end_dest,moved_xtekct_name,number_of_sectors)


    # creates the integrated images
    arr_of_end_points = np.linspace(number_of_images,0,number_of_sectors,dtype=int, endpoint=False)[::-1]
    in_sector = 0
    dim_matrix = np.zeros(get_image_array(os.path.join(source_dir, image_files[0])).shape)
    avg_matrix = dim_matrix
    counter = np.zeros(number_of_sectors)

    print(f"{number_of_images} images will be divided into {number_of_sectors} sectors the the last image number for each sector is then shown in this array: {arr_of_end_points}")
    for i in range(number_of_images):
        counter[in_sector]+=1
        avg_matrix += get_image_array(os.path.join(source_dir, image_files[i-1]))
        if arr_of_end_points[in_sector] == i+1:
            avg_matrix /= counter[in_sector]
            in_sector+=1
            matrix_to_tiff(np.uint16(avg_matrix), in_sector, end_dest)
            avg_matrix = dim_matrix
    f = open(os.path.join(end_dest,"info.txt"), "w")
    f.write(f"""The images is a result of adding up a discrete number of projections.\n For this case {number_of_images} images was divided into {number_of_sectors} sectors.\n The following array show how many images that was used per sector {counter} 
    """)







