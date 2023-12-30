import tensorflow as tf
import numpy as np

graph = tf.Graph()
with graph.as_default():
    # Placeholder for input sequence
    input_sequence = tf.placeholder(tf.float32, shape=[1, 10, 1], name='E_t')

    # Define the convolutional layer based on the provided snippet
    conv_layer = tf.layers.conv1d(input_sequence, filters=9, kernel_size=9, strides=1, padding='valid',
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None,
                                   name='conv00', data_format='channels_last', reuse=tf.AUTO_REUSE)
    
    # add summary operation to visualize the conv_layer output
    conv_summary = tf.summary.histogram('conv_layer_output', conv_layer)

    # merge all summaries
    merged_summary = tf.summary.merge_all()

# Initialize Tensorflow session and FileWriter for TensorBoard
with tf.Session(graph=graph) as sess:
    # Create a FileWriter to write summaries for TensorBoard
    writer = tf.summary.FileWriter('./logs', sess.graph)

    # Initialize global variable
    sess.run(tf.global_variables_initializer())

    # Generate a random input sequence for demonstration purposes
    random_sequence = np.random.rand(1, 10, 1)

    # Run the convolutional layer on the input sequence and collect the summary data\
    conv_output, summary = sess.run([conv_layer, merged_summary], feed_dict={input_sequence: random_sequence})

    # write summary data to the filewriter
    writer.add_summary(summary)

    print("output shape after convolution:", conv_output.shape)

    writer.close()

# Run visualize for logged infomation
# tensorboard --logdir=./logs