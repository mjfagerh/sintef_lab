import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from functools import lru_cache
from skimage.measure import label, regionprops
from matplotlib import colors
from scipy import ndimage
from numba import jit
"""
    task 
- skill på globale polaruavhengige feil, og polar feil
- velg gode slice for polar feil 
- 

"""

"""
        "Total relativ area difference":total_relative_area_dif,
        "local area difference": local_relative_area_dif,
        "number of islands": num_islands,
        "relative Volum difference": make_scalar_fit_pandas(relative_volum_dif, pd_size),
        "CM shift Volume": make_ndarray_fit_pandas(CM_sub - CM_mask, pd_size),
        "Delta major axis": delta_major,
        "Delta minor axis": delta_minor,
        "max_error": max_error,
        "num_error_preaks": num_error_peaks,
        "avg_error": avg_error
        
        
"""
name = ["50_proj","60_proj","70_proj","80_proj","85_proj","90_proj","100_proj","150_proj","300_proj","600_proj","1200_proj","3142_proj","mask"]
voxeldim = 0.0056707098990625

polar_slices = [i for i in range(0,182)]+[i for i in range(254,263)]+[i for i in range(799,1344)]+[i for i in range(1417,1490)]
faulty_slice = [i for i in range(530,545)]

data = []
for i in name:
    data.append(pd.read_csv(f'stats_{i}.csv'))

fgsize = (10,10)
colo = ['white','blue', 'red', 'yellow', 'orange', 'pink']
cmap = colors.ListedColormap(colo)
bounds=[0,1,2,3,4,5,6]
norm = colors.BoundaryNorm(bounds, cmap.N)
def get_npz(name):
    im = np.load(f"{name}.npz")["arr_0"]
    if name == "mask":
        return invert(im)
    return im

def get_mask_label(mask):
    mask_label, num1 = label(mask, connectivity=2, return_num=True)
    return mask_label, num1

def label_by_mask(mask, Asub):
    relabel,n = get_mask_label(mask)
    re_Asub = np.where(Asub != 0, relabel, Asub)
    return relabel, re_Asub
@lru_cache
def str2array(str,typ=float,delim = " "):
    if str == '0':
        return np.array([0])
    str = str.replace('[', '')
    str = str.replace(']', '')
    return np.fromstring(str, dtype=typ, sep=delim)

def get_index(pd):
    x = pd["Analysed slices index"]
    str_x = x[0]
    return str2array(str_x,typ=int,delim=",")

def get_first_val(pd,key,arr=True,typ=float):
    x = pd[key]
    str_x = x[0]
    if arr:
        return str2array(str_x,typ)
    else:
        return float(str_x)

def get_col_of_values(pd, key, indexes, typ=float):
    x = pd[key]
    x = np.take(x,indexes)
    arr = np.zeros(len(indexes))
    for i, el in enumerate(x):
        arr[i] = typ(el)
    return arr

def get_col_of_arr(pd, key, indexes, typ = float):
    x = pd[key].to_numpy()
    x = np.take(x, indexes)
    arr = np.zeros(len(indexes),dtype=object)
    maxi = 0
    j=0
    for i,el in enumerate(x):
        arr[i] = str2array(el)
        if maxi<np.max(str2array(el)):
            maxi = np.max(str2array(el))
            j = indexes[i]
    return arr

def sync_indexes(base_arr, input_arr):
    base_arr = np.array(base_arr)
    input_arr = np.array(input_arr)
    return np.intersect1d(base_arr, input_arr)

def get_global_analyzed_indexes(data,polar_slices=[]):
    base_arr = np.arange(0,1963)
    base_arr = np.delete(base_arr,faulty_slice)
    for pd in data:
        input_arr = get_index(pd)
        base_arr = sync_indexes(base_arr,input_arr)
    if len(polar_slices) != 0:
        base_arr = sync_indexes(base_arr, np.array(polar_slices))
    return base_arr

def flatten(arr):
    x = []
    for row in arr:
        for el in row:
            x.append(el)

    return np.array(x)

def get_CM(pd):
    x = pd["CM shift Volume"]
    x_str = x[0]
    return str2array(x_str)

def get_volume(pd):
    x = pd["relative Volum difference"]
    x_str = x[0]
    return round(x_str,2)

def avg_std(arr_flatt):
    return np.average(arr_flatt), np.std(arr_flatt)

def get_rel_area_dif(pd):
    x = pd["Total relativ area difference"]
    return np.sum(x)

def get_avg_std_val(data, key,indexes):
    std = []
    avg = []
    for i,pd in enumerate(data):
        col = get_col_of_values(pd,key,indexes)
        x = avg_std(col)
        avg.append(x[0])
        std.append(x[1])
    std = np.array(std)
    avg = np.array(avg)
    return avg, std

def get_avg_std_arr(data,key,indexes):
    std = []
    avg = []

    for i, pd in enumerate(data):

        col = get_col_of_arr(pd, key, indexes)


        col = flatten(col)


        x = avg_std(col)
        avg.append(x[0])
        std.append(x[1])
    std = np.array(std)
    avg = np.array(avg)
    return avg, std

def get_data_val(data,key,indexes):
    d = []
    for i, pd in enumerate(data):

        col = get_col_of_values(pd, key, indexes)


        d.append(col)
    return d

def get_data_arr(data,key,indexes):
    d = []
    for i, pd in enumerate(data):

        col = get_col_of_arr(pd, key, indexes)
        col = flatten(col)
        d.append(col)
    return d

def plotCM(data, name, plane='xy'):
    CMs = np.zeros((len(name),3))
    for i in range(len(data)):
        CMs[i] = get_CM(data[i])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    if plane == 'xy':
        x,y = CMs[:,0],CMs[:,1]
        limx = np.max(x)+1
        limy = np.max(y)+1
        for i,n in enumerate(name):
            ax.scatter(x[i],y[i], label=n)
        ax.set_xlabel('x', loc='right')
        ax.set_ylabel('y', loc='top', rotation=0)
        plt.xlim(-limx,limx)
        plt.ylim(-limy,limy)

    elif plane == 'xz':
        x, y = CMs[:, 0], CMs[:, 2]
        limx = np.max(x) + 1
        limy = np.max(y) + 1
        for i, n in enumerate(name):
            ax.scatter(x[i], y[i], label=n)
        ax.set_xlabel('x', loc='right')
        ax.set_ylabel('z', loc='top', rotation=0)
        plt.xlim(-limx, limx)
        plt.ylim(-limy, limy)
    elif plane == 'yz':
        x,y= CMs[:, 1], CMs[:, 2]
        limx = np.max(x) + 1
        limy = np.max(y) + 1
        for i, n in enumerate(name):
            ax.scatter(x[i], y[i], label=n)
        ax.set_xlabel('y', loc='right')
        ax.set_ylabel('z', loc='top', rotation=0)
        plt.xlim(-limx, limx)
        plt.ylim(-limy, limy)
    ax.grid()

    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')


    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.legend()
    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def plotVdif(data,name):
    V = [get_volume(i) for i in data]
    xtik = [i for i in range(len(name))]
    plt.figure()
    plt.plot(xtik, V)
    plt.xticks(xtik,name, fontsize=15)
    plt.yticks(V)
    plt.ylabel("$\\frac{V_{mask}-V}{V_{mask}}$", loc='top',rotation=0, fontsize=20)
    plt.grid()

def Megaplot(data,name):
    fig, ax = plt.subplots(2, 3, figsize=(30,25))
    x = np.array([i for i in range(len(name))])
    ind = [get_index(i) for i in data]
    for i in range(2):
        for j in range(3):
            ax[i,j].set_xticks(x,name)
            ax[i,j].grid()

    ax[0,0].title.set_text("Total relativ area difference")
    avg , std = get_avg_std_val(data,"Total relativ area difference",ind)
    ax[0,0].errorbar(x,avg , yerr=std,capsize=4)
    ax[0,0].set_ylabel("$\\frac{V_{mask}-V}{V_{mask}}$", loc='top',rotation=0)

    ax[0, 1].title.set_text("Number of islands")
    avg, std = get_avg_std_arr(data, "Number of islands", ind)
    ax[0, 1].errorbar(x,avg , yerr=std,capsize=4)
    ax[0, 1].set_ylabel("$\\frac{islands}{object}}$", loc='top', rotation=0)

    ax[0, 2].title.set_text("Relative Volum difference")
    y = [get_first_val(d, "Relative Volum difference", arr=False) for d in data]
    ax[0, 2].plot(x,y)
    ax[0, 2].set_ylabel("$\\frac{V_{mask}-V}{V_{mask}}$", loc='top', rotation=0)



    ax[1,0 ].title.set_text("Avg error")
    avg, std = get_avg_std_arr(data,"Avg error",ind)
    ax[1, 0].errorbar(x,avg*voxeldim , yerr=std*voxeldim,capsize=4)
    ax[1, 0].set_ylabel("mm", loc='top', rotation=0)

    ax[1, 1].title.set_text("max error")
    avg, std = get_avg_std_arr(data, "Max error", ind)
    ax[1, 1].errorbar(x, avg*voxeldim, yerr=std*voxeldim, capsize=4)
    ax[1, 1].set_ylabel("mm", loc='top', rotation=0)

    ax[1, 2].title.set_text("# error preaks")
    avg, std = get_avg_std_arr(data, "Num error peaks", ind)
    ax[1, 2].errorbar(x, avg, yerr=std, capsize=4)
    ax[1, 2].set_ylabel("number", loc='top', rotation=0)

    plt.savefig('total volum.pdf',format='pdf')

#avg value between objects
def get_value_per_object_in_slice(pd, name, slice):
    x = pd[name]
    values = str2array(x[slice])
    return values

def avg_between_listelement(arr):
    nrows = len(arr[0])
    avg_arr = np.zeros(nrows,dtype=object)
    for i in range(nrows):
        list_shape = arr[0][i].shape[0]
        base = np.zeros(list_shape)
        for j in range(len(arr)):
            base += arr[j][i]
        avg_arr[i] = base/len(arr)
    return avg_arr

def find_worst_reconstructed_area_slices(data, ind, N):
    l  = len(data)
    maxerr = np.zeros((l,N))
    slices = np.zeros((l,N))
    for reconstruction ,pd in enumerate(data):
        values = get_col_of_values(pd,"Total relativ area difference", ind)
        unsorted = list(zip(values, ind))
        sort = sorted(unsorted,key=lambda x: x[0])
        values, slic = list(zip(*sort))
        maxerr[reconstruction] = values[-10:]
        slices[reconstruction] = slic[-10:]
    return maxerr, slices

def find_biggest_value_arr(data,key,ind,):
    rec = 0
    maxi = np.zero
    i = 0
    for N, pd in enumerate(data):
        arr =get_col_of_arr(pd,key,ind)
        for j in range(len(arr)):
            m  = np.max(arr[j])
            if maxi < m:
                maxi = m
                i = j
                rec = N
    return rec, i, maxi

def collect_in_bins(r,err, bin_size):
    r = np.array(r)
    err = np.array(err)
    r0_avg = []
    r0_std = []
    err_avg = []
    err_std = []
    bind_dist = []
    bins = [i for i in range(bin_size,int(np.max(r)),bin_size)]
    bin_ind = np.digitize(r, bins)
    p = []

    N_bins = np.max(bin_ind)
    for i in range(N_bins+1):
        lab = np.where(bin_ind==i)
        bind_dist.append(len(lab[0]))
        rbin = np.take(r, lab)
        errbin = np.take(err, lab)
        r0_avg.append(np.average(rbin))
        r0_std.append(np.std(rbin))
        err_avg.append(np.average(errbin))
        err_std.append(np.std(errbin))

    return np.array(r0_avg), np.array(r0_std), np.array(err_avg), np.array(err_std), np.array(bind_dist)

def plot_global_error(data,name,tit,font,fig1,fig2,width,pad):
    fig, ax = plt.subplots(2, 1, figsize=(fig1,fig2))
    fig.tight_layout(pad=pad)
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13])
    ind = get_global_analyzed_indexes(data)
    for i in range(2):
        ax[i].set_xticks(x, [i.split("_")[0] for i in name[:-1]] + ["Ground truth"])
        ax[i].tick_params(axis="both", labelsize=font,color="black")
        ax[i].grid()
    mr, ar, trad, noi = get_slice_244(data)
    ax[0].set_title("Relative area difference a)", fontsize=tit, weight="bold")
    avg, std = get_avg_std_val(data,"Total relativ area difference", ind)
    #ax[0].errorbar(x, avg, yerr=std, capsize=4, color="black")
    ax[0].boxplot(get_data_val(data,"Total relativ area difference", ind),showmeans=True, showfliers=False, widths=width)
    ax[0].scatter(x,trad)
    ax[0].set_xticks([i for i in range(1,len(data)+1)], [i.split("_")[0] for i in name[:-1]] + ["Ground truth"])
    ax[0].set_ylabel("$\\frac{A_{GT}-A_{RS}}{A_{GT}}$", loc='center', rotation=0, fontsize=tit, labelpad=35)
    ax[0].set_xlabel("Number of projections used in the reconstruction", fontsize=font, loc="center")

    ax[1].set_title("Number of islands b)", fontsize=tit, weight="bold")
    #avg, std = get_avg_std_arr(data, "Number of islands", ind)
    #ax[1].errorbar(x, avg, yerr=std, capsize=4, color="black")
    ax[1].boxplot(get_data_arr(data, "Number of islands", ind),showmeans=True, showfliers=False, widths=width)
    ax[1].set_xticks([i for i in range(1,len(data)+1)], [i.split("_")[0] for i in name[:-1]] + ["Ground truth"])
    ax[1].set_ylabel("$\\frac{Islands}{Object}}$", loc='center', rotation=0, fontsize=tit, labelpad=30)
    ax[1].set_xlabel("Number of projections used in the reconstruction", fontsize=font, loc="center")
    ax[1].scatter(x, noi)

    """r = avg_between_listelement([get_col_of_arr(pd, "Radial distance", ind,typ=float) for pd in data])
    err = avg_between_listelement([get_col_of_arr(pd, "Local area difference", ind,typ=float) for pd in data])"""
    #not interested in perfect reconstructions therefore we use form proj 50-100 and take the avg error per object
    """radius_and_error = list(zip(flatten(r),flatten(err)))
    radius_and_error = sorted(radius_and_error, key=lambda x:x[0])
    r, err = list(zip(*radius_and_error))
    bin_size = 20
    r0_avg, r0_std, err_avg, err_std, bind_distribution = collect_in_bins(r,err,bin_size)"""
    """ax[2].title.set_text(f"Error vs radius, bin size = {bin_size*voxeldim} mm")
    ## if we want to incorperate error std wee need to look at radius bins
    ax[2].errorbar(r0_avg, err_avg,yerr=err_std)
    ax[2].set_ylabel("Relative error, averaged over all reconstructions from 50 to 100 projections")
    ax[2].set_xlabel(" Distance from rotation center [mm]")"""

    plt.savefig('total_volumbox.pdf', format='pdf', bbox_inches='tight')

def get_slice_244(data):
    small_obj_num = 0
    mr, ar, trad, noi =[],[],[],[]
    for i,pd in enumerate(data):
        mr.append(str2array(pd["Max error"].to_numpy()[244])[small_obj_num]*voxeldim)
        ar.append(str2array(pd["Avg error"].to_numpy()[244])[small_obj_num]*voxeldim)
        trad.append(str2array(pd["Local area difference"].to_numpy()[244])[small_obj_num])
        noi.append(str2array(pd["Number of islands"].to_numpy()[244])[small_obj_num])

    return mr,ar,trad,noi

def plot_polar_error(data,name,tit,font,fig1,fig2,width,pad):
    fig, ax = plt.subplots(2, 1, figsize=(fig1,fig2))
    fig.tight_layout(pad=pad)
    N=len(name)
    mr, ar, trad, noi = get_slice_244(data)
    # len name = 18
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13])
    ind = get_global_analyzed_indexes(data, polar_slices=polar_slices)
    for i in range(2):
        ax[i].grid()


    ax[0].set_title("Maximum error distance a)", fontsize=tit,weight="bold")
    avg, std = get_avg_std_arr(data, "Max error", ind)

    #ax[0].errorbar(x, avg*voxeldim, yerr=std*voxeldim, capsize=4,color="black")
    box_plot_data = get_data_arr(data, "Max error", ind)
    for d in range(len(box_plot_data)):
        for q in range(len(box_plot_data[d])):
            box_plot_data[d][q]*=voxeldim



    ax[0].boxplot(box_plot_data,showmeans=True, showfliers=False, widths=width,)
    ax[0].set_xticks([i for i in range(1,len(data)+1)], [i.split("_")[0] for i in name[:-1]] + ["Ground truth"], fontsize = font)

    ax[0].set_ylabel("$r_{max}$ [mm]", loc='top', rotation=0, fontsize=font)
    ax[0].set_xlabel("Number of projections used in the reconstruction", fontsize=font,loc="center")
    ax[0].tick_params(axis="both", labelsize=font)
    ax[0].scatter(x,mr)
    ax[1].set_title("Average error distance b)",fontsize=tit,weight="bold")
    #avg, std = get_avg_std_arr(data, "Avg error", ind)
    #ax[1].errorbar(x, avg*voxeldim, yerr=std*voxeldim, capsize=4,color="black")
    box_plot_data = get_data_arr(data, "Avg error", ind)
    for d in range(len(box_plot_data)):
        for q in range(len(box_plot_data[d])):
            box_plot_data[d][q] *= voxeldim
    ax[1].boxplot(box_plot_data,showmeans=True, showfliers=False, widths=width,)
    ax[1].set_xticks([i for i in range(1,len(data)+1)], [i.split("_")[0] for i in name[:-1]] + ["Ground truth"], fontsize=font)
    ax[1].scatter(x, mr)
    ax[1].set_ylabel("$\\bar{r}$ [mm]", loc='top', rotation=0, fontsize=font)
    ax[1].set_xlabel("Number of projections used in the reconstruction", fontsize=font, loc="center")
    ax[1].tick_params(axis="both", labelsize=font)
    """ax[2].title.set_text(f"Num error peaks")
    avg, std = get_avg_std_arr(data, "Num error peaks", ind)
    ax[2].errorbar(x, avg, yerr=std)
    ax[2].set_ylabel("#peaks")"""
    plt.savefig('polar_error_box.pdf', format='pdf', bbox_inches='tight')

def selected_slices(data,name,slices):
    fig, ax = plt.subplots(5, 3, figsize=(30,25))
    col =len(slices)
    x = [i for i in range(len(name))]
    for i in range(2,5):
        for j in range(3):
            ax[i, j].set_xticks(x, name)
            ax[i, j].grid()

    mask = np.load("../raw_files/mask.npz")['arr_0']
    mask = np.where(mask == 0, 1, 0)
    worst_image = np.load("../raw_files/50_proj.npz")['arr_0']
    ind = [get_index(i) for i in data]
    maxavg, std = get_avg_std_arr(data, "Max error", ind)
    avgavg, std = get_avg_std_arr(data, "Avg error", ind)
    areaavg, std = get_avg_std_val(data, "Total relativ area difference", ind)
    for i in range(col):

        reimage, reworst_image = label_by_mask(mask[:,:,slices[i]], worst_image[:,:,slices[i]])


        ax[0,i].imshow(reimage,interpolation='nearest', origin='lower',
                    cmap=cmap, norm=norm)


        ax[1,i].imshow(reworst_image,interpolation='nearest', origin='lower',
                    cmap=cmap, norm=norm)

    for i in range(col):

        ax[2, i].title.set_text(f"Avg error slice {slices[i]}")
        y = np.array([get_value_per_object_in_slice(j,'Avg error',slices[i]) for j in data])
        for j in range(len(y[0])):
            ax[2, i].plot(x, y[:,j]*voxeldim, color=colo[j+1], label=f"object {j}")
        ax[2,i].plot(x,avgavg*voxeldim,'--', color="black", label=f"Average of all analysed slices")
        ax[2, i].set_ylabel("mm", loc='top', rotation=0)
        ax[2,i].set_ylim(bottom = 0)
        ax[2, i].legend()


    for i in range(col):
        ax[3, i].title.set_text(f"Max error slice {slices[i]}")
        y = np.array([get_value_per_object_in_slice(j,'Max error',slices[i]) for j in data])
        for j in range(len(y[0])):
            ax[3, i].plot(x, y[:,j]*voxeldim, color=colo[j+1], label=f"object {j}")
        ax[3, i].plot(x, maxavg*voxeldim, '--', color="black", label=f"Average of all analysed slices")
        ax[3, i].set_ylabel("mm", loc='top', rotation=0)
        ax[3, i].set_ylim(bottom = 0)
        ax[3, i].legend()


    for i in range(col):
        ax[4, i].title.set_text(f"Average local area differnece {slices[i]}")
        y = np.array([get_value_per_object_in_slice(j,'Local area difference',slices[i]) for j in data])
        for j in range(len(y[0])):
            ax[4, i].plot(x, y[:,j], color=colo[j+1], label = f"object {j}")
        ax[4, i].plot(x, areaavg, '--', color="black", label=f"Average of all analysed slices")
        ax[4, i].set_ylabel(r"$\frac{A_{mask}-A}{A_{mask}}$", loc='top', rotation=0)
        ax[4, i].set_ylim(bottom = 0)
        ax[4,i].legend()
    plt.savefig('selected slices.pdf', format='pdf')
    plt.show()

    return 0

def get_mask():
    return np.load("../raw_files/mask.npz")['arr_0']
def invert(image):
    return np.where(image == 1, 0, 1)

def get_cm_image(image):
    x,y = ndimage.center_of_mass(image)
    return [int(y),int(x)]
@jit
def projxy(matrix):
    x, y, z = matrix.shape
    xy = np.zeros((x, y))
    for i in range(z):
        xy += matrix[:, :, i]
    return xy
@jit
def projxz(matrix):
    x, y, z = matrix.shape
    xz = np.zeros((x, z))
    for i in range(y):
        xz += matrix[:, i, :]
    return xz

def plotCM_3d(data, name):
    mask = invert(get_mask())
    xy = projxy(mask)
    xz = np.rot90(projxz(mask), axes=(1,0))
    fig, (ax1,ax2) = plt.subplots(1,2)
    im1 = ax1.imshow(xy, interpolation=None, origin='lower')
    im2 = ax2.imshow(xz, interpolation=None, origin='lower')

    midy, midx1 = get_cm_image(xy)
    midz, midx2 = get_cm_image(xz)
    ax1.add_patch(plt.Circle(get_cm_image(xy), color="black", radius=5, lw=3, fill=True))
    ax1.add_patch(plt.Circle(get_cm_image(xz), color="black", radius=5, lw=3, fill=True))
    for i in range(len(data)):
        x, y, z = get_CM(data[i]); x, y = int(x), int(y)
        ax1.add_patch(plt.Circle([z+midz, x+midx1],color="red", radius=1, lw=3, fill=True))
        ax2.add_patch(plt.Circle([y+midy, x+midx2], color="red", radius=1, lw=3, fill=True))

def r_diff_image(name):
    mask = invert(get_mask())
    diff_im = projxy(mask)
    x,y = diff_im.shape
    rotx = int(x/2)
    roty = int(y/2)
    fig, ax1 = plt.subplots(1, 1)
    for n in name:
        im = np.load(f"{n}.npz")["arr_0"]
        diff_im -= projxy(im)
    im1 = ax1.imshow(diff_im, interpolation = None, origin = 'lower')
    ax1.add_patch(plt.Circle([roty,rotx], color="black", radius=5, lw=3, fill=True))

def dist_points(x0,y0,x1,y1):
    dist = (np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2))
    if isinstance(dist, np.floating):
        return dist
    return min(dist)

def radius_map():
    mask = invert(get_mask()) #random slice
    x,y,z = mask.shape
    mask = invert(get_mask())[:,:,0]
    rmap = np.zeros(mask.shape)

    rx = int(y/2)
    ry = int(x/2)
    for i in range(x):
        for j in range(y):
            rmap[i,j] = dist_points(i,j,rx,ry)

    rmap0 = np.zeros((x,y,z))
    for i in range(z):
        rmap0[:,:,i] = rmap
        if i%20 ==0:
            print(i)


    np.savez("Radius_map",rmap0)
    return

def get_rmap():
    return np.load(f"../raw_files/Radius_map.npz")["arr_0"]

def plot_r(name,r_space):
    mask = invert(get_mask())
    x,y,z = mask.shape
    rmap = get_rmap()
    plt.figure(figsize = (15,10))
    percent_error = np.zeros(len(name), dtype=object)
    tit= 20
    for i,n in enumerate(name):
        im = get_npz(n)
        diff_im = (mask-im)*rmap
        mask1 = mask*rmap
        mask1.flatten()
        diff_array = diff_im.flatten()
        diff_array = np.sort(diff_array[diff_array != 0])
        maks1 = np.sort(mask1[mask1 !=0])
        maxr = max(maks1)
        count_mask = np.zeros(int(maxr/r_space)+1)
        count_diff = np.zeros(int(maxr/r_space)+1)
        area = np.zeros(int(maxr/r_space)+1)
        ble = np.array([i for i in range(0,int(maxr),r_space)])
        for j in range(len(ble)-1):
            area[j] = np.pi*(ble[j+1]**2-ble[j]**2)
            count_mask[j] = len(maks1[(ble[j] < maks1)&(maks1<ble[j+1])])
            count_diff[j] = len(diff_array[(ble[j] < diff_array) & (diff_array < ble[j+1])])
        count_mask[-1] = len(maks1[maxr>ble[-1]])
        count_diff[-1] = len(diff_array[ diff_array > ble[-1]])
        percent_error[i] = count_diff/(count_mask)
        d = n.split("_")[0]
        plt.plot((ble + (ble[1] - ble[0]) / 2)*voxeldim, count_mask)# label=n.split("_")[0]+" projections"
        break
    #count_mask = c_mask/(np.sum(c_mask))
    #plt.plot(ble[:-1] + (ble[1] - ble[0]) / 2, count_mask[:-1], label=n.split("_")[0]+" projections")
    """    
    avg_percent_error  = np.zeros(percent_error[0].shape)
    for i in range(len(name)):
        avg_percent_error += percent_error[i]
    avg_percent_error = avg_percent_error/len(name)
    """
    plt.legend()
    #plt.title("Radial error distribution", fontsize=20, weight = "bold")
    plt.xlabel("radius [mm]", fontsize=15)
    #plt.ylabel(r"$\frac{Error}{Voxels}$", loc='top', fontsize=20)
    plt.ylabel("Voxel count", loc='top', fontsize=15)
    plt.title(f"Radial distribution",fontsize=20, weight = "bold")
    plt.savefig("radial_distribution.pdf", bbox_inches='tight')

def plot_selected_slice(data,name, slices):
    mask = invert(np.load("../raw_files/mask.npz")['arr_0'])
    im1 = np.load(f"{name[0]}.npz")['arr_0']
    im2 = np.load(f"{name[1]}.npz")['arr_0']
    im3 = np.load(f"{name[2]}.npz")['arr_0']
    cols = ["Ground truth"] + [n.split("_")[0]+" projections" for n in name]
    rows = slices
    left, right, down, up = 90, 710, 90, 710
    nrow,ncol = 3,4
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(9, 9))


    for ax, col in zip(axes[0], cols):
        ax.set_title(col,weight="bold")

    for i in range(0,nrow-1):
        for j in range(0,ncol):
            axes[i,j].set_xticklabels([])
            if j == 0:
                continue
            axes[i,j].set_yticklabels([])
    for i in range(1,ncol):
        axes[2,i].set_yticklabels([])

    for i in range(1,ncol):
        yticks = axes[2,i].xaxis.get_major_ticks()
        print(yticks)
        yticks[-2].label1.set_visible(False)

    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel("slice "+str(row), rotation=90, size='large',weight="bold")

    for i in range(nrow):
        for j in range(ncol):
            axes[i,j].set_aspect('auto')

    fig.tight_layout()

    for i in range(3):
        mask_lab, im1_lab = label_by_mask(mask[:, :, slices[i]], im1[:, :, slices[i]])
        mask_lab, im2_lab = label_by_mask(mask[:, :, slices[i]], im2[:, :, slices[i]])
        mask_lab, im3_lab = label_by_mask(mask[:, :, slices[i]], im3[:, :, slices[i]])
        left,right, down, up  = 90,710,90,710
        axes[i, 0].imshow(mask_lab[left:right,down:up], interpolation='nearest', origin='lower',
                        cmap=cmap, norm=norm)
        axes[i, 1].imshow(im1_lab[left:right,down:up], interpolation='nearest', origin='lower',
                        cmap=cmap, norm=norm)
        axes[i, 2].imshow(im2_lab[left:right,down:up], interpolation='nearest', origin='lower',
                        cmap=cmap, norm=norm)
        axes[i, 3].imshow(im3_lab[left:right, down:up], interpolation='nearest', origin='lower',
                          cmap=cmap, norm=norm)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('Representative_slices.pdf', format='pdf')
    plt.show()

def find_worst_slice(data,name):
    ind = get_global_analyzed_indexes(data)

    maxe = 0
    w_slice = 0
    for i in range(5):
        tot_area = get_col_of_arr(data[i], "Local area difference", ind, typ=float)
        for j,el in enumerate(tot_area):
            if maxe<np.max(el):
                maxe = np.max(el)
                w_slice = ind[j]
        print(w_slice,maxe)

def area_distribution():
    mask = invert(get_mask())
    x, y, z = mask.shape
    area = []

    for i in range(z):
        mask_slice = mask[:, :, i]
        mask_label, num1 = label(mask_slice, connectivity=2, return_num=True)
        for a in regionprops(mask_label, cache=True):
             area.append(a.area*voxeldim**2)

        if i % 50 == 0:
            print(i)

    plt.figure(figsize = (15,10))
    y, binEdges = np.histogram(area, bins=100)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    smallest_po = mask[:, :, 244]

    mask_label, num1 = label(smallest_po, connectivity=2, return_num=True)
    smallest_area = regionprops(mask_label, cache=True)[0].area * voxeldim ** 2
    print(smallest_area)

    plt.plot([smallest_area,smallest_area],[0,max(y)], c="red")
    plt.plot(bincenters, y, '-', c='black')
    #plt.hist(area,bins=40)
    plt.title("Area distribution", fontsize=20, weight="bold")
    plt.xlabel("$mm^2$",fontsize=15)
    plt.ylabel("count",fontsize=15)
    plt.savefig("area_distribution.pdf",bbox_inches='tight')
    plt.show()
    return



["50_proj","55_proj","60_proj","65_proj","70_proj","75_proj","80_proj","85_proj","90_proj","95_proj","100_proj","150_proj","300_proj","600_proj","1200_proj","3142_proj","mask"]
#r_diff_image(["150_proj"])

#plotCM_3d(data, name)
tit=22
font=15
fig1 = 18
fig2 = 15
width = 0.3
pad = 7
print(get_slice_244(data))
#plot_polar_error(data,name,tit,font,fig1,fig2, width,pad)
#plot_global_error(data,name,tit,font,fig1,fig2, width,pad)

plot_r(["50_proj","60_proj","70_proj","80_proj","85_proj","90_proj","100_proj"], r_space=5)
#area_distribution()
#find_worst_slice(data,name)
#plot_selected_slice(data,["150_proj","85_proj","50_proj"], [244,752,1570])
#matrix_mask = get_npz("mask")


#plt.figure()
#plt.imshow(projxy(matrix_mask))
#plt.title("3142_proj")

#selected_slices(data,name,[245,600,1505])
#Megaplot(data,name)



