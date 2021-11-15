import numpy as np
from absl import logging

"""
Federated learning data distribution mapping functions
"""


def homogeneous(X, y, nendpoints, nclasses, rng):
    """Assign all data to all endpoints"""
    return [np.arange(len(y)) for _ in range(nendpoints)]


def extreme_heterogeneous(X, y, nendpoints, nclasses, rng):
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