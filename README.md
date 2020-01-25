# CNN-RNN-Yield-Prediction



This repository contains codes for the paper entitled <a href="https://www.frontiersin.org/articles/10.3389/fpls.2019.01750/full" target="_blank">"A CNN-RNN Framework for Crop Yield Prediction"</a> published in Frontiers in Plant Science Journal. The paper was authored by Saeed Khaki, Lizhi Wang, and Sotirios Archontoulis. In this paper, we proposed a framework for crop yield prediction.


### Please cite our paper if you use our code or data since it took a lot of time gathering and cleaning the data. Thanks!
```
@article{Khaki_2020,
   title={A CNN-RNN Framework for Crop Yield Prediction},
   volume={10},
   ISSN={1664-462X},
   url={http://dx.doi.org/10.3389/fpls.2019.01750},
   DOI={10.3389/fpls.2019.01750},
   journal={Frontiers in Plant Science},
   publisher={Frontiers Media SA},
   author={Khaki, Saeed and Wang, Lizhi and Archontoulis, Sotirios V.},
   year={2020},
   month={Jan}
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

 In the CSV files, they are named Wij, where i is the index of weather component and j is the week of year; i=1,...6, j=1,...,52. 
 
- `number of soil components measured at different depth (ns)`: 10 components.
- `number of planting date components (np)`: 16 (corn), 14 (soybean)
- `number of soil components measured at the surface (nss)`: 4

For detailed description of the variables, please read the data section of the paper.


## Downloading the Data

### The data analyzed in this study was obtained from different public data sources. Our team spent a lot of time cleaning and gathering the data. Please cite our papers if you use our data in your research.  

### To get the download link for the full version of the data, please send your requset to the following email:

skhaki@iastate.edu




## Success Strory of the Model in the Real World Application

We used the proposed method in practice in July 2019 for the prediction of the soybean yield of Iowa, Illinois, and Indiana states. This year soybean yield after harvesting published by USDA about two weeks ago. The actual yields are very close to what we predicted in July, which are as follows:

|State|         Predicted Yield in July 2019          |Actual Yield|
| ------------- |:-------------:| -----:|
|Iowa|                  53.44 (bu/ac)|                  53 (bu/ac)|
|Illinois|              48.71 (bu/ac)|                  51 (bu/ac)|
|Indiana|               47.01 (bu/ac)|                  49 (bu/ac)|



