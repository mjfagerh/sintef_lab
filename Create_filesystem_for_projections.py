import os
import shutil
import numpy as np



main_dest = "E:\Mats_08\CT-scanns\groundTrouth_TwoPoreSystem"
source_dir = "E:\Mats_08\CT-scanns\groundTrouth_TwoPoreSystem\\raw_data\\"
image_file_base = "KS20221003_MJHF_groundtruth"
list_of_N_projections = [50,55,60,65,70,75,80,85,90,95,100,150,300, 600,1200,3142]


def create_folder_for_images(number_of_images, dest=main_dest):
    dir_name = "{}_images".format(number_of_images)
    name = os.path.join(dest, dir_name)
    folder_exist = False
    if os.path.exists(name):
        print("dir does exist, therfore nothing will happen to it. You must delete it if you want to change it")
        folder_exist = True
    else:
        os.mkdir(name)
    return name, folder_exist


def get_fil_num(fil):
    return int(fil.split('.')[0].split('_')[-1])


def get_fil_type(fil):
    return fil.split('.')[-1]


def get_raw_files():
    return os.listdir(source_dir)


def get_num_images():
    fil_names = get_raw_files()
    larges_num = 0
    for fil in fil_names:
        if get_fil_type(fil) == 'tif':
            n = get_fil_num(fil)
            if n > larges_num:
                larges_num = n
    return larges_num


def get_filenames_of_equal_spaced_images(number_of_images):
    N = get_num_images()
    index = np.linspace(1, N, number_of_images, dtype=int)
    output_files = []
    input_files = get_raw_files()
    for input_fil in input_files:
        if (get_fil_type(input_fil) != 'xtekct'):
            if get_fil_num(input_fil) in index:
                output_files.append(input_fil)
    return output_files


def move_xtekct_fil(dir_path):
    """

    :param dir_path: the path of the new folder
    :return: None
    """
    fil_type = 'xtekct'
    for fil in get_raw_files():
        if get_fil_type(fil) == fil_type:
            shutil.copy2(source_dir+fil, dir_path+'/'+'0'+fil)
            return "\\"+'0'+fil
    return None


def fix_image_num(dir_path):
    files = os.listdir(dir_path)
    i = 1
    for fil in files:
        if get_fil_type(fil) == 'tif':
            str_num = str(i).zfill(4)
            base, typ = fil.split('.')
            new_fil = '_'.join(base.split('_')[:-1])+'_'+str_num+'.'+typ
            i += 1
            os.rename(os.path.join(dir_path, fil), os.path.join(dir_path, new_fil))


def adjust_xtekct_fil(dir_path, fil_name, projections):
    """

    :param dir_path: this is the path to the new folder
    :param fil_name: this is the name of the xtec file that has been moved
    :param projections: this is the number of projections that is in the new file
    :return: none
    """
    with open(dir_path+fil_name, mode='r') as fil:
        new_file = []
        list_lines = fil.read().split('\n')
        for arg in list_lines:
            if "Projections" in arg:
                new_arg = "Projections={}".format(projections)
                new_file.append(new_arg)
            elif "AngularStep" in arg:
                new_arg = "AngularStep={:.7f}".format(360/(projections))
                new_file.append(new_arg)
            else:
                new_file.append(arg)
        input = '\n'.join(new_file)
    with open(dir_path+fil_name, mode='w') as fil:
        fil.write(input)




def Create_filesystem_for_projections(N_to_move):
    dir_path, folder_exist = create_folder_for_images(N_to_move)
    if not folder_exist:
        input_files = get_filenames_of_equal_spaced_images(N_to_move)
        for fil in input_files:
            shutil.copy2(source_dir+fil, dir_path)
        moved_xtekct_name = move_xtekct_fil(dir_path)
        fix_image_num(dir_path)
        adjust_xtekct_fil(dir_path,moved_xtekct_name, N_to_move)


def main():
    for N_to_move in list_of_N_projections:
        Create_filesystem_for_projections(N_to_move)

if __name__=="__main__":
    main()