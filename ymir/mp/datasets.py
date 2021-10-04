import sklearn.datasets as skds
import sklearn.preprocessing as skp
import numpy as np

import abc


class DataIter:
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
    def __init__(self, X, y):
        self.X, self.y = X, y
        self.classes = np.unique(self.y).shape[0]

    @abc.abstractmethod
    def train(self) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abc.abstractmethod
    def test(self) -> tuple[np.ndarray, np.ndarray]:
        pass

    def get_iter(self, split, batch_size=None, filter=None, map=None) -> DataIter:
        X, y = self.train() if split == 'train' else self.test()
        X, y = X.copy(), y.copy()
        if filter is not None:
            idx = filter(y)
            X, y = X[idx], y[idx]
        if map is not None:
            X, y = map(X, y)
        return DataIter(X, y, batch_size)
    
    @abc.abstractmethod
    def fed_split(self, batch_sizes: np.ndarray, iid: bool) -> dict[int, DataIter]:
        pass


class MNIST(Dataset):
    def __init__(self):
        X, y = skds.fetch_openml('mnist_784', return_X_y=True)
        X, y = X.to_numpy(), y.to_numpy()
        X = skp.MinMaxScaler().fit_transform(X)
        X = X.astype(np.float32)
        # X = X.reshape(-1, 28, 28, 1).astype(np.float32)
        y = skp.LabelEncoder().fit_transform(y).astype(np.int8)
        super().__init__(X, y)
        
    def train(self):
        return self.X[:60000], self.y[:60000]
    
    def test(self):
        return self.X[60000:], self.y[60000:]

    def fed_split(self, batch_sizes, iid):
        filter = (lambda i: lambda y: y == i % self.classes) if not iid else (lambda _: None)
        return {i: self.get_iter("train", b, filter=filter(i)) for i, b in enumerate(batch_sizes)}


class CIFAR10(Dataset):
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
        filter = (lambda i: lambda y: y == i % self.classes) if not iid else (lambda _: None)
        return {i: self.get_iter("train", b, filter=filter(i)) for i, b in enumerate(batch_sizes)}


class KDDCup99(Dataset):
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
        filter = (lambda i: lambda y: (y == (i + 1 if i >= 11 else i) % self.classes) | (y == 11)) if not iid else (lambda _: None)
        return {i: self.get_iter("train", b, filter=filter(i)) for i, b in enumerate(batch_sizes)}