from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import optax

from . import adversaries
from ymir import mp


"""
Endpoint side functionalities
"""


class Collaborator:
    def __init__(self, opt, opt_state, loss, data, epochs):
        self.opt_state = opt_state
        self.data = data
        self.batch_size = data.batch_size
        self.epochs = epochs
        # having the following for each client makes it a bit slower (but substantially faster overall) for the first round due to jit
        self.update = update(opt, loss)


def update(opt, loss):
    """Do some local training and return the gradient"""
    @jax.jit
    def _apply(params, opt_state, X, y):
        grads = jax.grad(loss)(params, X, y)
        updates, opt_state = opt.update(grads, opt_state, params)
        return grads, opt_state, updates
    return _apply