import sys

import jax
import jax.numpy as jnp


@jax.jit
def euclid_dist(a, b):
    return jnp.sqrt(jnp.sum((a - b)**2, axis=-1))

def unzero(x):
    return max(x, sys.float_info.epsilon)