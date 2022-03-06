"""
A client for the Ditto FL personalization algorithm proposed in `https://arxiv.org/abs/2012.04221 <https://arxiv.org/abs/2012.04221>`_
"""

from functools import partial

import jax
import optax

import ymir.path

from . import scout


class Scout(scout.Scout):
    """A federated learning client which performs personalization according to the Ditto algorithm."""

    def __init__(self, params, opt, opt_state, loss, data, epochs, lamb=0.1):
        """
        Constructor for a Ditto Scout.

        Arguments:
        - params: initial model parameters
        - opt: optimizer to use for training
        - opt_state: initial optimizer state
        - loss: loss function to use for training
        - data: data to use for training
        - epochs: number of epochs to train for per round
        - lamb: lambda parameter for the Ditto algorithm
        """
        super().__init__(opt, opt_state, loss, data, epochs)
        self.params = params
        self.local_opt_state = opt_state
        self.local_update = partial(update, opt, loss)
        self.lamb = lamb

    def step(self, params, return_weights=False):
        """
        Perform a single local training loop.

        Arguments:
        - params: the parameters of the global model from the most recent round
        - return_weights: if True, return the weights of the clients else return the gradients from the local training
        """
        p, self.opt_state = self.update(params, self.opt_state, *next(self.data))
        for _ in range(self.epochs):
            self.params, self.opt_state = self.local_update(
                self.params, params, self.opt_state, self.lamb, *next(self.data)
            )
        return p if return_weights else ymir.path.tree.sub(params, p)


@partial(jax.jit, static_argnums=(
    0,
    1,
))
def update(opt, loss, local_params, global_params, opt_state, lamb, X, y):
    """
    Local learning step.

    Arguments:
    - opt: optimizer
    - loss: loss function
    - local_params: local model parameters
    - global_params: global model parameters
    - opt_state: optimizer state
    - lamb: lambda parameter, scales the amount of regularization
    - X: samples
    - y: labels
    """
    grads = ymir.path.tree.add(
        jax.grad(loss)(local_params, X, y), ymir.path.tree.scale(ymir.path.tree.sub(local_params, global_params), lamb)
    )
    updates, opt_state = opt.update(grads, opt_state, local_params)
    local_params = optax.apply_updates(local_params, updates)
    return local_params, opt_state
