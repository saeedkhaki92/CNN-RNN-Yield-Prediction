# CNN-RNN-Yield-Prediction



This repository contains codes for the paper entitled "A CNN-RNN Framework for Crop Yield Prediction". The paper was authored by Saeed Khaki, Lizhi Wang, and Sotirios Archontoulis. In this paper, we proposed a framework for crop yield prediction.



## Getting Started 

Please install the following packages in Python3:

- numpy
- tensorflow
- matplotlib


## Dimension of Input Data

Let nw, ns, np, and nss be the number of weather components, soil components meaured at different depth, planting time component, and soil components meaured at the surface. Let m be the total number of observations. So `X` is `m-by-(nw+ns+np+nss)`. We added three columns to the begining of the matrix `X` which are location_id, year, and yield response variable.  If input data is not in this format, the code would not run.

- `number of weather components (nw)`: 6 components meaured over 52 weeks.
- `number of soil components measured at different depth (ns)`: 10 components meaured at 10 different depths of soil.
- `number of planting date components (np)`: 16 (corn), 14 (soybean)
- `number of soil components measured at the surface (nss)`: 4


## The CNN-RNN Hyperparameters

We used the following hyperparameters to train the CNN-RNN model. The W-CNN and S-CNN models
both have 4 convolutional layers. In the CNN models, downsampling was performed by average pooling with stride of 2. The output of W-CNN is followed by a fully-connected layer which has 60 neurons for corn yield prediction and 40 neurons for soybean
yield prediction. The output of the S-CNN model is followed by a fully-connected layer which has 40
neurons. The RNN layer has a time length of 5 years since we considered a 5-year yield dependencies. The
RNN layer has LSTM cells with 64 hidden units. After trying different network designs, we found this
architecture to provide the best overall performance.

All weights were initialized with the Xavier method (Glorot and Bengio, 2010). We used stochastic
gradient decent (SGD) with mini-batch size of 25. Adam optimizer (Kingma and Ba, 2014) were used
with the learning rate of 0.03% which was divided by 2 every 60,000 iterations. The model was trained for
the maximum 350,000 iterations. We used rectified linear unit (ReLU) activation function for the CNNs
and FC layer. The output layer had a linear activation function.
