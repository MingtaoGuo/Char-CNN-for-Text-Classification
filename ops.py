import tensorflow as tf


def conv(name, inputs, nums_out, k_h, k_w):
    nums_in = int(inputs.shape[-1])
    with tf.variable_scope(name):
        W = tf.get_variable("W", [k_h, k_w, nums_in, nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.]))
        inputs = tf.nn.conv2d(inputs, W, [1, 1, 1, 1], "VALID") + b
    return inputs

def max_pooling(inputs, k_h, k_w, strides):
    return tf.nn.max_pool(inputs, [1, k_h, k_w, 1], [1, strides, 1, 1], "VALID")

def relu(inputs):
    return tf.nn.relu(inputs)

def dropout(inputs, keep_prob):
    return tf.nn.dropout(inputs, keep_prob)

def fully_connected(name, inputs, nums_out):
    inputs = tf.layers.flatten(inputs)
    with tf.variable_scope(name):
        nums_in = int(inputs.shape[-1])
        W = tf.get_variable("W", [nums_in, nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.]))
        inputs = tf.matmul(inputs, W) + b
    return inputs