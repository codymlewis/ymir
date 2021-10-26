import jax
import numpy as np

"""
General utility library for ymir
"""


@jax.jit
def tree_uniform(tree, low=0.0, high=1.0):
    """Create an equivalently shaped tree with random number elements in the range [low, high)"""
    return jax.tree_map(lambda x: np.random.uniform(low=low, high=high, size=x.shape), tree)


@jax.jit
def tree_add_normal(tree, loc=0.0, scale=1.0):
    """Add normally distributed noise to each element of the tree, (mu=loc, sigma=scale)"""
    return jax.tree_map(lambda x: x + np.random.normal(loc=loc, scale=scale, size=x.shape), tree)


@jax.jit
def tree_mul(tree, scale):
    """Multiply the elements of a pytree by the value of scale"""
    return jax.tree_map(lambda x: x * scale, tree)


@jax.jit
def tree_add(*trees):
    """Element-wise add any number of pytrees"""
    return jax.tree_multimap(lambda *xs: sum(xs), *trees)