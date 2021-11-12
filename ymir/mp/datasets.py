import sklearn.datasets as skds
import sklearn.preprocessing as skp
import numpy as np
import os
from absl import logging

import abc

import datalib

"""
Load and preprocess datasets
"""


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


def homogeneous(X, y, nendpoints, nclasses, rng):
    """Assign all data to all endpoints"""
    return [np.arange(len(y)) for _ in range(nendpoints)]

def heterogeneous(X, y, nendpoints, nclasses, rng):
    """Assign each endpoint only the data from each class"""
    return [np.isin(y, i % nclasses) for i in range(nendpoints)]

def lda(X, y, nendpoints, nclasses, rng):
    """Latent dirichlet allocation from https://arxiv.org/abs/2002.06440"""
    distribution = [[] for _ in range(nendpoints)]
    proportions = rng.dirichlet(np.repeat(0.5, nendpoints), size=nclasses)
    for c in range(nclasses):
        idx_c = np.where(y == c)[0]
        dists_c = np.split(idx_c, np.round(np.cumsum(proportions[c]) * len(idx_c)).astype(int)[:-1])
        distribution = [distribution[i] + d.tolist() for i, d in enumerate(dists_c)]
    logging.debug(f"distribution: {proportions.tolist()}")
    return distribution


def load(dataset, dir="data"):
    fn = f"{dir}/{dataset}.npz"
    if not os.path.exists(fn):
        datalib.download(dir, dataset)
    ds = np.load(f"{dir}/{dataset}.npz")
    X, y, train = ds['X'], ds['y'], ds['train']
    return Dataset(X, y, train)