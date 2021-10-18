import sklearn.datasets as skds
import sklearn.preprocessing as skp
import numpy as np
import os

import abc


"""
Load and preprocess datasets
"""


class DataIter:
    """Iterator that gives random batchs in pairs of (sample, label)"""
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = y.shape[0] if batch_size is None else min(batch_size, y.shape[0])
        self.idx = np.arange(y.shape[0])

    def __iter__(self):
        return self

    def __next__(self):
        idx = np.random.choice(self.idx, self.batch_size, replace=False)
        return self.X[idx], self.y[idx]


class Dataset:
    """Abstract definition for the datasets"""
    def __init__(self, X, y, train):
        self.X, self.y, self.train_idx = X, y, train
        self.classes = np.unique(self.y).shape[0]

    def train(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the training subset"""
        return self.X[self.train_idx], self.y[self.train_idx]

    def test(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the testing subset"""
        return self.X[~self.train_idx], self.y[~self.train_idx]

    def get_iter(self, split, batch_size=None, filter=None, map=None) -> DataIter:
        """Generate an iterator out of the dataset"""
        X, y = self.train() if split == 'train' else self.test()
        X, y = X.copy(), y.copy()
        if filter is not None:
            idx = filter(y)
            X, y = X[idx], y[idx]
        if map is not None:
            X, y = map(X, y)
        return DataIter(X, y, batch_size)
    
    def fed_split(self, batch_sizes, mappings):
        """Divide the dataset for federated learning"""
        return [self.get_iter("train", b, filter=lambda y: np.isin(y, m)) for b, m in zip(batch_sizes, mappings)]


def load(dataset, dir="~/ymir_datasets"):
    dir = os.path.expanduser(dir)
    ds = np.load(f"{dir}/{dataset}.npz")
    X, y, train = ds['X'], ds['y'], ds['train']
    return Dataset(X, y, train)


class MNIST(Dataset):
    """The basic MNIST dataset"""
    def __init__(self):
        X, y = skds.fetch_openml('mnist_784', return_X_y=True)
        X, y = X.to_numpy(), y.to_numpy()
        X = skp.MinMaxScaler().fit_transform(X)
        X = X.reshape(-1, 28, 28, 1).astype(np.float32)
        y = skp.LabelEncoder().fit_transform(y).astype(np.int8)
        super().__init__(X, y)
        
    def train(self):
        return self.X[:60000], self.y[:60000]
    
    def test(self):
        return self.X[60000:], self.y[60000:]

    def fed_split(self, batch_sizes, iid):
        """Split as one class per endpoint of not iid otherwise everyone gets a copy of the full set"""
        filter = (lambda i: lambda y: y == i % self.classes) if not iid else (lambda _: None)
        return [self.get_iter("train", b, filter=filter(i)) for i, b in enumerate(batch_sizes)]


class CIFAR10(Dataset):
    """The CIFAR 10 tiny image object detection dataset"""
    def __init__(self):
        X, y = skds.fetch_openml('CIFAR_10', return_X_y=True)
        X, y = X.to_numpy(), y.to_numpy()
        X = skp.MinMaxScaler().fit_transform(X)
        X = X.reshape(-1, 32, 32, 3).astype(np.float32)
        y = skp.LabelEncoder().fit_transform(y).astype(np.int8)
        super().__init__(X, y)
        
    def train(self):
        return self.X[:50000], self.y[:50000]
    
    def test(self):
        return self.X[50000:], self.y[50000:]

    def fed_split(self, batch_sizes, iid):
        """Split as one class per endpoint of not iid otherwise everyone gets a copy of the full set"""
        filter = (lambda i: lambda y: y == i % self.classes) if not iid else (lambda _: None)
        return [self.get_iter("train", b, filter=filter(i)) for i, b in enumerate(batch_sizes)]


class KDDCup99(Dataset):
    """The KDD Cup '99 intrusion detection system dataset"""
    def __init__(self):
        X, y = skds.fetch_kddcup99(shuffle=False, return_X_y=True)
        # remove the classes not in the test set
        idx = (y != b'spy.') & (y != b'warezclient.')
        X, y = X[idx], y[idx]
        y = (le := skp.LabelEncoder()).fit_transform(y).astype(np.int8)
        for i in [1, 2, 3]:
            X[:, i] = skp.LabelEncoder().fit_transform(X[:, i])
        X = skp.MinMaxScaler().fit_transform(X)
        X = X.astype(np.float32)
        super().__init__(X, y)
    
    def train(self):
        return self.X[:345815], self.y[:345815]
    
    def test(self):
        return self.X[345815:], self.y[345815:]

    def fed_split(self, batch_sizes, iid):
        """Give everyone normal class data and one other classed data if non iid otherwise everyone gets a copy of the full set"""
        filter = (lambda i: lambda y: (y == (i + 1 if i >= 11 else i) % self.classes) | (y == 11)) if not iid else (lambda _: None)
        return [self.get_iter("train", b, filter=filter(i)) for i, b in enumerate(batch_sizes)]