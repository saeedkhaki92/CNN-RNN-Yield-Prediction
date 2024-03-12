import tensorflow as tf
import numpy as np

input_data = np.random.rand(6, 6, 3).astype(np.float32)

conv_layer = tf.layers.conv1d(inputs=input_data, 
                              filters=3, 
                              kernel_size=(3, 3),
                              strides=1,
                              padding='valid', 
                              activation=tf.nn.relu, 
                              use_bias=True)

output = conv_layer

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    output_value = sess.run(output)

    print("Input data:")
    print(input_data.shape)
    print("\n output after convolution")
    print(output_value.shape)
