import sys
import os
import itertools
import sklearn.metrics as skm

import jax
import jax.numpy as jnp
import numpy as np

import datalib
import ymir


@jax.jit
def euclid_dist(a, b):
    return jnp.sqrt(jnp.sum((a - b)**2, axis=-1))

def unzero(x):
    return max(x, sys.float_info.epsilon)


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


def accuracy(net, **_):
    """Find the accuracy of the models predictions on the data"""
    @jax.jit
    def _apply(params, X, y):
        return jnp.mean(jnp.argmax(net.apply(params, X), axis=-1) == y)
    return _apply

def asr(net, attack_from, attack_to, **_):
    """Find the success rate of a label flipping/backdoor attack that attempts the mapping attack_from -> attack_to"""
    @jax.jit
    def _apply(params, X, y):
        preds = jnp.argmax(net.apply(params, X), axis=-1)
        idx = y == attack_from
        return jnp.sum(jnp.where(idx, preds, -1) == attack_to) / jnp.sum(idx)
    return _apply


class Neurometer:
    """Measure aspects of the model"""
    def __init__(self, net, datasets, evals, add_keys=[], **kwargs):
        self.datasets = datasets
        self.evaluators = {e: globals()[e](net, **kwargs) for e in evals}
        self.results = {f"{d} {e}": [] for d, e in itertools.product(datasets.keys(), evals)}
        for k in add_keys:
            self.results[k] = []

    def add_record(self, params):
        """Add a measurement of the chosen aspects with respect to the current params, return the latest results"""
        for ds_type, ds in self.datasets.items():
            for eval_type, eval in self.evaluators.items():
                self.results[f"{ds_type} {eval_type}"].append(eval(params, *next(ds)))
        return {k: v[-1] for k, v in self.results.items()}

    def add(self, key, value):
        """Add a single result to the results"""
        self.results[key].append(value)

    def get_results(self):
        """Return overall results formatted into jax.numpy arrays"""
        for k, v in self.results.items():
            self.results[k] = jnp.array(v)
        return self.results