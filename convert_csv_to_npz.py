import numpy as np
import pandas as pd



data_csv=pd.read_csv('./corn_samples.csv',delimiter=',')

data_npz=np.array(data_csv)

np.savez_compressed('./corn_samples_npz',data=data_npz)



