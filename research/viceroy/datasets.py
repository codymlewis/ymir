import os

import numpy as np

import datalib
import ymir

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

    def get_iter(self, split, batch_size=None, filter=None, map=None, rng=np.random.default_rng()):
        """Generate an iterator out of the dataset"""
        X, y = self.train() if split == 'train' else self.test()
        X, y = X.copy(), y.copy()
        if filter is not None:
            idx = filter(y)
            X, y = X[idx], y[idx]
        if map is not None:
            X, y = map(X, y)
        return ymir.mp.datasets.DataIter(X, y, batch_size, self.classes, rng)
    
    def fed_split(self, batch_sizes, mappings=None):
        """Divide the dataset for federated learning"""
        if mappings is not None:
            return [self.get_iter("train", b, filter=lambda y: np.isin(y, m)) for b, m in zip(batch_sizes, mappings)]
        return [self.get_iter("train", b) for b in batch_sizes]


def load(dataset, dir="data"):
    fn = f"{dir}/{dataset}.npz"
    if not os.path.exists(fn):
        datalib.download(dir, dataset)
    ds = np.load(f"{dir}/{dataset}.npz")
    X, y, train = ds['X'], ds['y'], ds['train']
    return Dataset(X, y, train)