# CNN-RNN-Yield-Prediction



This repository contains codes for the paper entitled <a href="https://arxiv.org/abs/1911.09045" target="_blank">"A CNN-RNN Framework for Crop Yield Prediction"</a> published in <a href="https://www.frontiersin.org/articles/10.3389/fpls.2019.01750/abstract" target="_blank"> Frontiers in Plant Science Journal</a> . The paper was authored by Saeed Khaki, Lizhi Wang, and Sotirios Archontoulis. In this paper, we proposed a framework for crop yield prediction.



## Getting Started 

Please install the following packages in Python3:

- numpy
- tensorflow
- matplotlib


## Dimension of Input Data

Let nw, ns, np, and nss be the number of weather components, soil components meaured at different depth, planting time component, and soil components meaured at the surface. Let m be the total number of observations. So `X` is `m-by-(nw+ns+np+nss)`. We added three columns to the begining of the matrix `X` which are location_id, year, and yield response variable.  If input data is not in this format, the code would not run.

- `number of weather components (nw)`: 6 components.
- `number of soil components measured at different depth (ns)`: 10 components.
- `number of planting date components (np)`: 16 (corn), 14 (soybean)
- `number of soil components measured at the surface (nss)`: 4


## Data Availability Statement

The data analyzed in this study was obtained from different public data sources.

