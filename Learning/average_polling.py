import tensorflow as tf

# Define input data
input_data = tf.placeholder(tf.float32, shape=[None, 5, 3])  # Shape: (batch_size, sequence_length, num_channels)

# Apply AveragePooling1D
pooling_layer = tf.keras.layers.AveragePooling1D(pool_size=5, strides=1, padding='valid')

# Apply pooling operation
output = pooling_layer(input_data)

# Print output shape
print(output.shape)  # Output shape: (None, 5, 3)
