import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


class Pore:
    def __init__(self, matrix, voxelvolum=1):
        """"
        :param matrix: the pure matrix collected from avizo, 1 should represent porespace
        :param voxelsize: units of micronL
        """
        self.body = matrix
        self.hight = matrix.shape[2]
        self.area_dim = matrix.shape[:2]
        self.pore_volume = np.count_nonzero(self.body)
        self.voxel_volume = voxelvolum #should be in mL
        self.simple_model = self.create_simplified_model()

    def get_slice(self, z):
        return self.body[:, :, z]

    def get_total_area_of_slice(self, z):
        slice = self.get_slice(z)
        return np.count_nonzero(slice)

    def create_simplified_model(self):
        simple_model = np.zeros(self.hight)
        for i in range(self.hight):
            simple_model[i] = self.get_total_area_of_slice(i)
        est_vol = np.sum(simple_model)
        print(f"estimated volume of object is {est_vol*self.voxel_volume} mL")
        return simple_model

    def run_flow_experiment(self, pumprate):
        simple_model = self.simple_model.copy()
        print(simple_model)
        np.savetxt("simple_model.csv",simple_model)


        """
        :param pumprate: units of micronliter [mu L/sek]
        :return: time series of hight that is filled upp
        """
        time_development = [["time", "slice"]]  # [sek]
        sek = 0
        N_voxels_to_fill_per_timeunit = pumprate / self.voxel_volume
        print(f"number of voxel that is filled per sekund {N_voxels_to_fill_per_timeunit}")
        voxels_left_to_fill = N_voxels_to_fill_per_timeunit

        run = True
        slice = 0
        c=0
        while run:

            if slice == self.hight:
                run = False
                break

            voxels_to_fill_in_slice = simple_model[slice]
            #print(f"voxels_to_fill_in_slice={voxels_to_fill_in_slice}")
            #print(f"voxels_left_to_fill = {voxels_left_to_fill}")
            if voxels_to_fill_in_slice < voxels_left_to_fill:
                voxels_left_to_fill -= simple_model[slice]
                simple_model[slice] = 0
                slice += 1

                continue

            elif voxels_to_fill_in_slice == voxels_left_to_fill:
                simple_model[slice] = 0
                sek += 1
                time_development.append([sek, slice+c])
                slice += 1
                voxels_left_to_fill = N_voxels_to_fill_per_timeunit

                continue

            elif voxels_to_fill_in_slice > voxels_left_to_fill:
                simple_model[slice] -= voxels_left_to_fill
                sek += 1
                time_development.append([sek, slice+c])
                voxels_left_to_fill = N_voxels_to_fill_per_timeunit

                continue

        return np.array(time_development)

def invert(image):
    return np.where(image == 0, 1 ,0)

def perMin_to_perSek(flowrate):
    return flowrate/60
def get_mask():
    return np.load("../raw_files/mask.npz")['arr_0'][:,:,:]
pore_matrix = invert(get_mask())
print(f"total number of pore voxels {np.count_nonzero(pore_matrix)}")

voxelsize = (0.0056**3)*1e-3 #mL
print(f"voxel volume {voxelsize} mL")
flowrate = perMin_to_perSek(0.006) #mL/sek
print(f"flowrate {flowrate} mL/s")
experiment = Pore(pore_matrix, voxelsize)
list = np.asarray(experiment.run_flow_experiment(flowrate))
print(np.asarray(list,dtype=object))
np.savetxt("experiment.csv",list,fmt="%s")