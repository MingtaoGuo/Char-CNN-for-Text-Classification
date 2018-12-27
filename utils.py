import numpy as np
import pandas as pd

def get_char2id(all_char="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "):
    char_list = list(all_char)
    char2id = {}
    for idx, char in enumerate(char_list):
        char2id[char] = idx
    char_set = set(char_list)
    return char2id, char_set


def read(path="./dataset/train.csv", batch_size=128, seq_size = 1014, nums_class=4):
    char2id, char_set = get_char2id()
    chunks = pd.read_csv(path, encoding="latin1")
    chunk = pd.DataFrame(chunks).sample(batch_size)
    chunk = np.array(chunk)
    batch = np.ones([batch_size, seq_size]) * char2id[' ']
    label = np.zeros([batch_size, nums_class])
    for i in range(batch_size):
        str_ = list(chunk[i][1] + "." + chunk[i][2])
        label[i, chunk[i][0] - 1] = 1
        for seq_i, char in enumerate(str_):
            if char not in char_set:
                batch[i, seq_i] = char2id[' ']
            else:
                batch[i, seq_i] = char2id[char]
    return batch, label

def char2data(path="./dataset/test.csv", seq_size = 1014, nums_class=4):
    char2id, char_set = get_char2id()
    chunks = pd.read_csv(path, encoding="latin1")
    chunk = pd.DataFrame(chunks)
    chunk = np.array(chunk)
    nums_data = int(chunk.shape[0])
    testdata = np.ones([nums_data, seq_size]) * char2id[' ']
    label = np.zeros([nums_data, nums_class])
    for i in range(nums_data):
        str_ = list(chunk[i][1] + "." + chunk[i][2])
        label[i, chunk[i][0] - 1] = 1
        for seq_i, char in enumerate(str_):
            if char not in char_set:
                testdata[i, seq_i] = char2id[' ']
            else:
                testdata[i, seq_i] = char2id[char]
    return testdata, label

