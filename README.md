# CNN-RNN-Yield-Prediction



This repository contains codes for the paper entitled <a href="https://arxiv.org/abs/1911.09045" target="_blank">"A CNN-RNN Framework for Crop Yield Prediction"</a> published in <a href="https://www.frontiersin.org/articles/10.3389/fpls.2019.01750/abstract" target="_blank"> Frontiers in Plant Science Journal</a> . The paper was authored by Saeed Khaki, Lizhi Wang, and Sotirios Archontoulis. In this paper, we proposed a framework for crop yield prediction.


### Please cite our paper if you use our code or data since it took a lot of time gathering and cleaning the data. Thanks!
```
@article{khaki2019cnn,
  title={A CNN-RNN Framework for Crop Yield Prediction},
  author={Khaki, Saeed and Wang, Lizhi and Archontoulis, Sotirios V},
  journal={arXiv preprint arXiv:1911.09045},
  year={2019}
}


@article{khaki2019crop,
  title={Crop yield prediction using deep neural networks},
  author={Khaki, Saeed and Wang, Lizhi},
  journal={Frontiers in plant science},
  volume={10},
  year={2019},
  publisher={Frontiers Media SA}
}


```


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

For detail description of the variables, please read the data section of the paper.


## Downloading the Data

### The data analyzed in this study was obtained from different public data sources. Our team spent a lot of time cleaning and gathering the data. Please cite our papers if you use our data in your research.  

### To download the corn data use the following link:


### To download the soybean data use the following link:






