# CNN-RNN-Yield-Prediction



This repository contains codes for the paper entitled "A CNN-RNN Framework for Crop Yield Prediction". The paper was authored by Saeed Khaki, Lizhi Wang, and Sotirios Archontoulis. In this paper, we proposed a framework for crop yield prediction.



## Getting Started 

Please install the following packages in Python3:

- numpy
- tensorflow
- matplotlib
- Pandas


## Dimension of Input Data

Let nw, ns, np, and nss be the number of weather components, soil components meaured at different depth, planting time component, and soil components meaured at the surface. Let m be the total number of observations. So `X` is `m-by-(nw+ns+np+nss)`. We added three columns to the begining of the matrix `X` which are location_id, year, and yield response variable.  If input data is not in this format, the code would not run.

-`nw`: 52*6
-`ns`: 10 *10
-`np`: 16 (corn), 14 (soybean)
-`nss`: 4
