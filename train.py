from network import Char_CNN
import tensorflow as tf
from utils import read



def train(train_path, model_path, batch_size, seq_size, vec_size, nums_class, nums_char, drop_rate, learning_rate, max_itr):
    inputs = tf.placeholder(tf.int32, [None, seq_size])
    labels = tf.placeholder(tf.float32, [None, nums_class])
    keep_prob = tf.placeholder(tf.float32)
    embedding = tf.get_variable("embedding", [nums_char, vec_size], initializer=tf.truncated_normal_initializer(stddev=0.02), trainable=False)
    char_vec = tf.nn.embedding_lookup(embedding, inputs)
    char_vec = tf.expand_dims(char_vec, [-1])
    pred = Char_CNN(char_vec, keep_prob)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), tf.argmax(labels, axis=1)), dtype=tf.float32))
    loss = -tf.reduce_mean(tf.log(tf.reduce_sum(pred * labels, axis=1) + 1e-10))
    Opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for i in range(max_itr):
        BATCH, LABELS = read(train_path, batch_size, seq_size, nums_class)
        sess.run(Opt, feed_dict={inputs: BATCH, labels: LABELS, keep_prob: drop_rate})
        if i % 10 == 0:
            [LOSS, ACCURACY] = sess.run([loss, accuracy], feed_dict={inputs: BATCH, labels: LABELS, keep_prob: 1.0})
            print("Iteration: %d, Loss: %f, Training acc: %f"%(i, LOSS, ACCURACY))
        if i % 100 == 0:
            saver.save(sess, model_path + "model.ckpt")
        pass


