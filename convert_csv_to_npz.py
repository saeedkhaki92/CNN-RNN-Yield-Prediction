import numpy as np
import pandas as pd



data_csv=pd.read_csv('./name_of_your_file.csv',delimiter=',')

data_npz=np.array(data_csv)

np.savez_compressed('./name_of_your_file',data=data_npz)



