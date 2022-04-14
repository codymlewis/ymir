"""
General functions for dealing with model weights.
"""

import numpy as np


def uniform(weights, low=0.0, high=1.0, rng=np.random.default_rng()):
    """Create an equivalently shaped tree with random number elements in the range [low, high)"""
    return [rng.uniform(low=low, high=high, size=x.shape) for x in weights]


def add_normal(weights, loc=0.0, scale=1.0, rng=np.random.default_rng()):
    """Add normally distributed noise to each element of the tree, (mu=loc, sigma=scale)"""
    return [x + rng.normal(loc=loc, scale=scale, size=x.shape) for x in weights]


def mul(weights_a, weights_b):
    """Multiply the elements of two pytrees"""
    return [a * b for a, b in zip(weights_a, weights_b)]


def div(weights_a, weights_b):
    """Divide the elements of two pytrees"""
    return [a / b for a, b in zip(weights_a, weights_b)]


def scale(weights, scale):
    """Multiply the elements of a pytree by the value of scale"""
    return [x * scale for x in weights]


def add(*weights):
    """Element-wise add any number of pytrees"""
    return [sum(xs) for xs in zip(*weights)]


def sub(weights_a, weights_b):
    """Subtract tree_b from tree_a"""
    return [a - b for a, b in zip(weights_a, weights_b)]


def ravel(weights):
    """Flatten weights into a vector"""
    return np.concatenate([np.ravel(x) for x in weights])


def unravel(weights, skeleton):
    """Split the weights into a tree with the specified skeleton"""
    i = 0
    unravelled_weights = []
    for shape, length in skeleton:
        unravelled_weights.append(np.reshape(weights[i:i + length], shape))
        i += length
    return unravelled_weights


def skeleton(weights):
    """Return the shape of the weights"""
    return [(x.shape, np.prod(x.shape)) for x in weights]


def get_names(model):
    return [w.name for w in model.weights]


def minimum(weights, val):
    """Element-wise minimum of the weights and a scalar"""
    return [np.minimum(x, val) for x in weights]


def maximum(weights, val):
    """Element-wise maximum of the weights and a scalar"""
    return [np.maximum(x, val) for x in weights]
