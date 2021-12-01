import jax
import jax.numpy as jnp
import numpy as np

from . import server


"""
Static norm clipping aggregator.
"""


class Server(server.AggServer):
    def __init__(self, params, opt, opt_state, network, rng, C=1.0, M=1.0):
        super().__init__(params, opt, opt_state, network, rng)
        self.M = M
        self.C = C
        self.K = len(network)
    
    def update(self, all_grads):
        pass

    def scale(self, all_grads):
        G = jnp.array([jax.flatten_util.ravel_pytree(g)[0] for g in all_grads])
        return jax.vmap(lambda g: 1 / jnp.maximum(1, jnp.linalg.norm(g, ord=2) / self.M))(G)