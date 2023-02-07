import numpy as np
from skimage.measure import label
import matplotlib.pyplot as plt
from scipy import ndimage


def invert(image):
    return np.where(image == 0, 1 ,0)

def qplot(image):
    plt.figure()
    plt.imshow(image)
    plt.show()

def get_mask():
    return np.load("../raw_files/mask.npz")['arr_0']

def get_GroundTruth():
    return np.load("GroundTruth.npz")['arr_0']


def fill(area):
    smallest_allowed_area = 200
    area = invert(area)
    relabel = label(area, connectivity=2)
    objects = ndimage.find_objects(relabel)
    i = 0
    for obj in objects:
        xshape, yshape = relabel[obj].shape
        label_value = relabel[obj].flatten()[0]
        if i == 0: # by creation of find_object the first object is always the outer square
            relabel = np.where(relabel == label_value, 0, relabel)
            i+=1
        if xshape * yshape < smallest_allowed_area: #removes unwanted pixel errors
            print("found pxl")
            relabel = np.where(relabel == label_value, 0, relabel)

    return np.where(relabel !=0 , 0 , 1) # makes the image binary and flipps the values

def fill_edges(mask):
    x_shape, y_shape, z_shape = mask.shape
    for z in range(z_shape):
        if z % 10 == 0:
            print('slice {} of {}'.format(z, z_shape))
        mask[:, :, z] = fill(mask[:, :, z])
    return mask

def create_mask(GT):
    copy = GT.copy()
    mask = fill_edges(copy)
    np.savez("../raw_files/mask.npz", mask)


def main():
    return 0

def fill_top_bot(GT,min,lower_boundary,higer_boundary, max):
    for i in range(min, lower_boundary):
        GT[:, :, i] = 1
    for i in range(higer_boundary, max):
        GT[:, :, i] = 1
    return GT
if __name__ == "__main__":
    GT = np.load('../raw_files/GT.npz')['arr_0']
    GT = fill_top_bot(GT,0,135,1830,1962)
    create_mask(GT)















