from network import Char_CNN
import tensorflow as tf
from utils import char2data



def test(test_path, model_path, seq_size, nums_class, nums_char, vec_size):
    inputs = tf.placeholder(tf.int32, [None, seq_size])
    labels = tf.placeholder(tf.float32, [None, nums_class])
    keep_prob = tf.placeholder(tf.float32)
    embedding = tf.get_variable("embedding", [nums_char, vec_size], initializer=tf.truncated_normal_initializer(stddev=0.02), trainable=False)
    char_vec = tf.nn.embedding_lookup(embedding, inputs)
    char_vec = tf.expand_dims(char_vec, [-1])
    pred = Char_CNN(char_vec, keep_prob)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), tf.argmax(labels, axis=1)), dtype=tf.float32))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, model_path + ".\\model.ckpt")
    testdata, testlabel = char2data(test_path, seq_size, nums_class)
    c = 0
    for i in range(testdata.shape[0]):
        c += sess.run(accuracy, feed_dict={inputs: testdata[i:i+1], labels: testlabel[i:i+1], keep_prob: 1.0})
    test_acc = c / testdata.shape[0]
    print("Test accuracy: %f"%(test_acc))

# if __name__ == "__main__":
#     test("./dataset/test.csv", "./save_para/", 1014, 4, 69, 128)