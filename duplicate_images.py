import os
import shutil
from Code.Create_filesystem_for_projections import get_fil_type, adjust_xtekct_fil

N_dupli = 40
num_image = 50
N_proj = num_image*N_dupli
source = r"D:\Mats_08\CT-scanns\groundTrouth_TwoPoreSystem\{}_images".format(num_image)
new_dir = os.path.join(source, "duplicate")


def make_copies():
    fils = os.listdir(source)
    name = "duplicate"
    path_dir = os.path.join(source, name)
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.mkdir(new_dir)
    j = 0
    for fil in fils:
        if get_fil_type(fil) == "tif":
            fil_path = os.path.join(source, fil)
            for i in range(1,41):
                str_num = str(40*j+i).zfill(4)
                base, typ = fil.split('.')
                new_fil = '_'.join(base.split('_')[:-1]) + '_' + str_num + '.' + typ
                shutil.copy(fil_path, os.path.join(new_dir,new_fil))
            j+=1

def move_xtekct_fil():
    for fil in os.listdir(source):
        if get_fil_type(fil) == 'xtekct':
            shutil.copy2(os.path.join(source,fil), new_dir+'/0'+fil)
            print(new_dir+'0'+fil)
            return "/"+'0'+fil
    return None



make_copies()
xfil = move_xtekct_fil()
adjust_xtekct_fil(new_dir,xfil,N_proj)
