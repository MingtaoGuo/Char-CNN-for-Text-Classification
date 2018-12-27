from ops import *

def Char_CNN(inputs, keep_prob, is_small=True):#inputs: [None, 1014, 128, 1]
    if is_small:
        feat_len = 256 #Small: 256, Large: 1024
        fc_nums = 1024
    else:
        feat_len = 1024  # Small: 256, Large: 1024
        fc_nums = 2048
    inputs = relu(conv("conv1", inputs, feat_len, k_h=7, k_w=128))
    inputs = tf.transpose(inputs, [0, 1, 3, 2])
    inputs = max_pooling(inputs, k_h=2, k_w=1, strides=3)
    inputs = relu(conv("conv2", inputs, feat_len, k_h=7, k_w=feat_len))
    inputs = tf.transpose(inputs, [0, 1, 3, 2])
    inputs = max_pooling(inputs, k_h=2, k_w=1, strides=3)
    inputs = relu(conv("conv3", inputs, feat_len, k_h=3, k_w=feat_len))
    inputs = tf.transpose(inputs, [0, 1, 3, 2])
    inputs = relu(conv("conv4", inputs, feat_len, k_h=3, k_w=feat_len))
    inputs = tf.transpose(inputs, [0, 1, 3, 2])
    inputs = relu(conv("conv5", inputs, feat_len, k_h=3, k_w=feat_len))
    inputs = tf.transpose(inputs, [0, 1, 3, 2])
    inputs = relu(conv("conv6", inputs, feat_len, k_h=3, k_w=feat_len))
    inputs = tf.transpose(inputs, [0, 1, 3, 2])
    inputs = max_pooling(inputs, k_h=2, k_w=1, strides=3)
    inputs = relu(dropout(fully_connected("fc1", inputs, fc_nums), keep_prob))
    inputs = relu(dropout(fully_connected("fc2", inputs, fc_nums), keep_prob))
    logits = fully_connected("output", inputs, 4)
    pred = tf.nn.softmax(logits)
    return pred


if __name__ == "__main__":
    inputs = tf.placeholder(tf.float32, [None, 1014, 70, 1])
    keep_prob = tf.placeholder(tf.float32)
    Char_CNN(inputs, keep_prob)