from dataclasses import dataclass
from typing import Mapping

import numpy as np
import jax
import optax

from . import adversaries
from . import compression
from .. import mp


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
    data: mp.datasets.DataIter
    batch_size: int

    def __init__(self, opt_state, data):
        self.opt_state = opt_state
        self.data = data
        self.batch_size = data.batch_size
