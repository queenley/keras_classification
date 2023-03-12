import argparse


def make_parser():
    parser = argparse.ArgumentParser("Image Classifier with Keras")

    parser.add_argument("--data_path", required=True, type=str, help="the raw data path")
    parser.add_argument("--batch_size", default=90, type=int, help="batch size")
    parser.add_argument("--img_size", default=(256, 256), help="the image size")
    parser.add_argument("--ckpt_path", default="./save_ckpt/", type=str, help="the path of checkpoint")
    parser.add_argument("--train_learning_rate", type=float, help="the learning rate for training")
    parser.add_argument("--tune_learning_rate", type=float, help="the learning rate for tuning")
    parser.add_argument("--num_epochs", type=int, help="the number epoch")
