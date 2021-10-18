import argparse
import os

import sklearn.datasets as skds
import sklearn.preprocessing as skp
import numpy as np


def load():
    return skds.fetch_kddcup99(return_X_y=True)


def preprocess(X, y):
    idx = (y != b'spy.') & (y != b'warezclient.')
    X, y = X[idx], y[idx]
    y = skp.LabelEncoder().fit_transform(y).astype(np.int8)
    for i in [1, 2, 3]:
        X[:, i] = skp.LabelEncoder().fit_transform(X[:, i])
    X = skp.MinMaxScaler().fit_transform(X)
    X = X.astype(np.float32)
    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download, preprocess and save the KDD Cup '99 dataset.")
    parser.add_argument('file', metavar='FILE', type=str, nargs='?', default='~/ymir_datasets/kddcup99',
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
    np.savez_compressed(fn, X=X, y=y, train=(np.arange(len(y)) < 345_815))