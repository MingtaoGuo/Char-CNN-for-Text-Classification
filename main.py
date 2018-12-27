import argparse
from train import train
from test import test

IS_TRAINING = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seq_size", type=int, default=1014)
    parser.add_argument("--vec_size", type=int, default=128)
    parser.add_argument("--nums_class", type=int, default=4)
    parser.add_argument("--nums_char", type=int, default=69)
    parser.add_argument("--max_itr", type=int, default=100000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--drop_rate", type=float, default=0.5)
    parser.add_argument("--model_path", type=str, default="./save_para/")
    parser.add_argument("--training_path", type=str, default="./dataset/train.csv")
    parser.add_argument("--testing_path", type=str, default="./dataset/test.csv")

    args = parser.parse_args()

    if IS_TRAINING:
        train(args.training_path, args.model_path, args.batch_size, args.seq_size, args.vec_size, args.nums_class, args.nums_char, args.drop_rate, args.learning_rate, args.max_itr)
    else:
        test(args.testing_path, args.model_path, args.seq_size, args.nums_class, args.nums_char, args.vec_size)

