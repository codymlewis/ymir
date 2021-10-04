from dataclasses import dataclass
from typing import Mapping

import numpy as np
import jax
import optax

from . import adversaries
from . import compression


"""
Client functions
"""


def update(opt, loss):
    """Do some local training and return the gradient"""
    @jax.jit
    def _apply(params, opt_state, X, y):
        grads = jax.grad(loss)(params, X, y)
        _, opt_state = opt.update(grads, opt_state)
        return grads, opt_state
    return _apply


@dataclass
class Client:
    """Store client-based data"""
    opt_state: optax.OptState
    data: Mapping[str, np.ndarray]
    batch_size: int