from abc import ABC, abstractmethod

import jax
import numpy as np
import optax


class Server(ABC):

    def __init__(self, network, params, opt=optax.sgd(1), rng=np.random.default_rng()):
        self.network = network
        self.params = params
        ravelled_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
        self.params_len = len(ravelled_params)
        self.unraveller = jax.jit(unravel_fn)
        self.opt = opt
        self.opt_state = self.opt.init(params)
        self.updater = updater(opt)
        self.rng = rng

    def update(self, grads):
        self.params, self.opt_state = self.updater(self.params, self.unraveller(grads), self.opt_state)

    @abstractmethod
    def step(self):
        pass


def updater(opt):

    @jax.jit
    def _apply(params, grads, opt_state):
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state

    return _apply
