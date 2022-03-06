"""
General pytree related functions.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnums=(
    1,
    2,
    3,
))
def uniform(tree, low=0.0, high=1.0, rng=np.random.default_rng()):
    """Create an equivalently shaped tree with random number elements in the range [low, high)"""
    return jax.tree_map(lambda x: rng.uniform(low=low, high=high, size=x.shape), tree)


@partial(jax.jit, static_argnums=(
    1,
    2,
    3,
))
def add_normal(tree, loc=0.0, scale=1.0, rng=np.random.default_rng()):
    """Add normally distributed noise to each element of the tree, (mu=loc, sigma=scale)"""
    return jax.tree_map(lambda x: x + rng.normal(loc=loc, scale=scale, size=x.shape), tree)


@jax.jit
def mul(tree_a, tree_b):
    """Multiply the elements of two pytrees"""
    return jax.tree_map(lambda a, b: a * b, tree_a, tree_b)


@jax.jit
def div(tree_a, tree_b):
    """Divide the elements of two pytrees"""
    return jax.tree_map(lambda a, b: a / b, tree_a, tree_b)


@jax.jit
def scale(tree, scale):
    """Multiply the elements of a pytree by the value of scale"""
    return jax.tree_map(lambda x: x * scale, tree)


@jax.jit
def add(*trees):
    """Element-wise add any number of pytrees"""
    return jax.tree_multimap(lambda *xs: sum(xs), *trees)


@jax.jit
def sub(tree_a, tree_b):
    """Subtract tree_b from tree_a"""
    return jax.tree_map(lambda a, b: a - b, tree_a, tree_b)


@jax.jit
def flatten(tree):
    """Flatten a pytree into a vector"""
    return jax.flatten_util.ravel_pytree(tree)[0]


@partial(jax.jit, static_argnums=(1, ))
def minimum(tree, val):
    """Multiply the elements of two pytrees"""
    return jax.tree_map(lambda x: jnp.minimum(x, val), tree)


@partial(jax.jit, static_argnums=(1, ))
def maximum(tree, val):
    """Multiply the elements of two pytrees"""
    return jax.tree_map(lambda x: jnp.maximum(x, val), tree)
