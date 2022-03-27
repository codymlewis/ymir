"""
Standard client collaborators for federated learning.
"""

from functools import partial

import jax
import optax

import ymir.path


class Scout:
    """A client for federated learning, holds its own data and personal learning variables."""

    def __init__(self, opt, opt_state, loss, data, epochs, backend=None):
        """
        Constructor for a Scout.

        Arguments:
        - opt: optimizer to use for training
        - opt_state: initial optimizer state
        - loss: loss function to use for training
        - data: data to use for training
        - epochs: number of epochs to train for per round
        - backend: backend to use for training, as in cpu, gpu, etc.
        """
        self.opt_state = opt_state
        self.data = data
        self.batch_size = data.batch_size
        self.epochs = epochs
        self.opt = opt
        self.loss = loss
        self.backend = backend
        self.update = partial(jax.jit(update, static_argnums=(
            0,
            1,
        ), backend=backend), opt, loss)

    def step(self, params, return_weights=False):
        """
        Perform a single local training loop.

        Arguments:
        - params: the parameters of the global model from the most recent round
        - return_weights: if True, return the weights of the clients else return the gradients from the local training
        """
        p = params
        for _ in range(self.epochs):
            p, self.opt_state = self.update(p, self.opt_state, *next(self.data))
        return p if return_weights else ymir.path.tree.sub(params, p)


def update(opt, loss, params, opt_state, X, y):
    """
    Local learning step.

    Arguments:
    - opt: optimizer
    - loss: loss function
    - params: model parameters
    - opt_state: optimizer state
    - X: samples
    - y: labels
    """
    grads = jax.grad(loss)(params, X, y)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state
