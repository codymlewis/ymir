"""
Load a dataset, handle the subset distribution, and provide an iterator.
"""

from typing import Union

import numpy as np


class HFDataIter:
    """Iterator that gives random batchs in pairs of $(X_i, y_i) : i \subseteq {1, \ldots, N}$"""

    def __init__(self, ds, batch_size, classes, rng):
        self.ds = ds
        self.batch_size = len(ds) if batch_size is None else min(batch_size, len(ds))
        self.len = len(ds)
        self.classes = classes
        self.rng = rng

    def __iter__(self):
        """Return this as an iterator."""
        return self

    def filter(self, filter_fn):
        self.ds = self.ds.filter(filter_fn)
        self.len = len(self.ds)
        return self

    def map(self, map_fn):
        self.ds = self.ds.map(map_fn)
        self.len = len(self.Y)
        return self

    def __next__(self):
        """Get a random batch."""
        idx = self.rng.choice(self.len, self.batch_size, replace=False)
        return self.ds[idx]['X'], self.ds[idx]['Y']

    def __len__(self):
        return len(self.ds)


class DataIter:
    """Iterator that gives random batchs in pairs of $(X_i, y_i) : i \subseteq {1, \ldots, N}$"""

    def __init__(self, X, Y, batch_size, classes, rng):
        """
        Construct a data iterator.
        
        Arguments:
        - X: the samples
        - y: the labels
        - batch_size: the batch size
        - classes: the number of classes
        - rng: the random number generator
        """
        self.X, self.Y = X, Y
        self.batch_size = len(Y) if batch_size is None else min(batch_size, len(Y))
        self.len = len(Y)
        self.classes = classes
        self.rng = rng

    def filter(self, filter_fn):
        idx = filter_fn(self.Y)
        self.X, self.Y = self.X[idx], self.Y[idx]
        self.len = len(self.Y)
        return self

    def map(self, map_fn):
        self.X, self.Y = map_fn(self.X, self.Y)
        self.len = len(self.Y)
        return self

    def __iter__(self):
        """Return this as an iterator."""
        return self

    def __next__(self):
        """Get a random batch."""
        idx = self.rng.choice(self.len, self.batch_size, replace=False)
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.ds)


class Dataset:
    """Object that contains the full dataset, primarily to prevent the need for reloading for each client."""

    def __init__(self, ds):
        """
        Construct the dataset.
        Arguments:
        - ds: a hugging face dataset
        """
        self.ds = ds
        self.classes = len(np.union1d(np.unique(ds['train']['Y']), np.unique(ds['test']['Y'])))
        self.input_shape = ds['train'][0]['X'].shape

    def get_iter(
        self,
        split,
        batch_size=None,
        idx=None,
        filter_fn=None,
        map_fn=None,
        in_memory=True,
        rng=np.random.default_rng()
    ) -> Union[DataIter, HFDataIter]:
        """
        Generate an iterator out of the dataset.
        
        Arguments:
        - split: the split to use, either "train" or "test"
        - batch_size: the batch size
        - idx: the indices to use
        - filter: a function that takes the labels and returns whether to keep the sample
        - map: a function that takes the samples and labels and returns a subset of the samples and labels
        - rng: the random number generator
        """
        if filter_fn is not None:
            ds.filter(filter_fn)
        if map_fn is not None:
            ds.map(map_fn)
        if in_memory:
            X, Y = self.ds[split]['X'], self.ds[split]['Y']
            if idx is not None:
                X, Y = X[idx], Y[idx]
            return DataIter(X, Y, batch_size, self.classes, rng)
        ds = self.ds[split]
        if idx is not None:
            ds = ds.select(idx)
        return HFDataIter(ds, batch_size, self.classes, rng)

    def fed_split(self, batch_sizes, mapping=None, in_memory=True, rng=np.random.default_rng()):
        """
        Divide the dataset for federated learning.
        
        Arguments:
        - batch_sizes: the batch sizes for each client
        - mapping: a function that takes the dataset information and returns the indices for each client
        - rng: the random number generator
        """
        if mapping is not None:
            distribution = mapping(self.ds['train']['Y'], len(batch_sizes), self.classes, rng)
            return [
                self.get_iter("train", b, idx=d, in_memory=in_memory, rng=rng)
                for b, d in zip(batch_sizes, distribution)
            ]
        return [self.get_iter("train", b, in_memory=in_memory, rng=rng) for b in batch_sizes]
