import numpy as np
import os

import datalib

"""
Load and preprocess datasets
"""


def load(dataset, dir="data"):
    fn = f"{dir}/{dataset}.npz"
    if not os.path.exists(fn):
        datalib.download(dir, dataset)
    ds = np.load(f"{dir}/{dataset}.npz")
    X, y, train = ds['X'], ds['y'], ds['train']
    return Dataset(X, y, train)


class DataIter:
    """Iterator that gives random batchs in pairs of (sample, label)"""
    def __init__(self, X, y, batch_size, classes, rng):
        self.X = X
        self.y = y
        self.batch_size = y.shape[0] if batch_size is None else min(batch_size, y.shape[0])
        self.idx = np.arange(y.shape[0])
        self.classes = classes
        self.rng = rng

    def __iter__(self):
        return self

    def __next__(self):
        idx = self.rng.choice(self.idx, self.batch_size, replace=False)
        return self.X[idx], self.y[idx]


class Dataset:
    def __init__(self, X, y, train):
        self.X, self.y, self.train_idx = X, y, train
        self.classes = np.unique(self.y).shape[0]

    def train(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the training subset"""
        return self.X[self.train_idx], self.y[self.train_idx]

    def test(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the testing subset"""
        return self.X[~self.train_idx], self.y[~self.train_idx]

    def get_iter(self, split, batch_size=None, idx=None, filter=None, map=None, rng=np.random.default_rng()) -> DataIter:
        """Generate an iterator out of the dataset"""
        X, y = self.train() if split == 'train' else self.test()
        X, y = X.copy(), y.copy()
        if idx is not None:
            X, y = X[idx], y[idx]
        if filter is not None:
            fidx = filter(y)
            X, y = X[fidx], y[fidx]
        if map is not None:
            X, y = map(X, y)
        return DataIter(X, y, batch_size, self.classes, rng)
    
    def fed_split(self, batch_sizes, mapping=None, rng=np.random.default_rng()):
        """Divide the dataset for federated learning"""
        if mapping is not None:
            distribution = mapping(*self.train(), len(batch_sizes), self.classes, rng)
            return [self.get_iter("train", b, idx=d, rng=rng) for b, d in zip(batch_sizes, distribution)]
        return [self.get_iter("train", b, rng=rng) for b in batch_sizes]