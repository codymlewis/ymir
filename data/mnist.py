import argparse
import os

import sklearn.datasets as skds
import sklearn.preprocessing as skp
import numpy as np


def load():
    X, y = skds.fetch_openml('mnist_784', return_X_y=True)
    return X.to_numpy(), y.to_numpy()


def preprocess(X, y):
    X = skp.MinMaxScaler().fit_transform(X)
    X = X.reshape(-1, 28, 28, 1).astype(np.float32)
    y = skp.LabelEncoder().fit_transform(y).astype(np.int8)
    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download, preprocess and save the MNIST dataset.')
    parser.add_argument('file', metavar='FILE', type=str, nargs='?', default='~/ymir_datasets/mnist',
                    help='File to save to')
    args = parser.parse_args()
    fn = os.path.expanduser(args.file)
    dir = os.path.dirname(fn)

    print("Downloading data...")
    X, y = load()
    print("Done. Preprocessing data...")
    X, y = preprocess(X, y)
    print(f"Done. Saving as a compressed file to {fn}")
    os.makedirs(dir, exist_ok=True)
    np.savez_compressed(fn, X=X, y=y, train=(np.arange(len(y)) < 60_000))