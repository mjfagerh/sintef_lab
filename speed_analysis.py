import numpy as np
import pandas as pd
csv_name = "scanner_data_15_feb"


def add_scann(Filname="_",Error="_",Kv=0,uA=0,W=0,ExposureTime=0,Gain=0,Binning="_"):
    row = pd.DataFrame([[Filname, Error, Kv, uA, W, ExposureTime, Gain, Binning]])
    row.to_csv(csv_name,mode='a',index=False, header=False)

def remove_empty_scanns():
    df = pd.read_csv("scanner_data_15_feb")
    df =df.drop(df[df.Filename == "_"].index)
    df.to_csv(csv_name, index=False)



def calulate_volume_change():
    mask = np.where(np.load("../raw_files/mask.npz")['arr_0']==1,0,1)
    sample = np.load(filname)["arr_0"]
    volume_mask = np.sum(mask)
    volume_sample = np.sum(sample)
    Error = (volume_mask - volume_sample)/volume_mask
    return Error



