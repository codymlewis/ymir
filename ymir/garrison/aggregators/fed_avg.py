import jax
import jax.numpy as jnp
import numpy as np
from absl import logging

from . import server


"""
Basic federated averaging proposed in https://arxiv.org/abs/1602.05629
"""


class Server(server.AggServer):
    def __init__(self, params, network, C=1.0):
        self.batch_sizes = jnp.array([c.batch_size * c.epochs for c in network.clients])
        self.network = network
        self.C = C
        self.K = len(network)
    
    def update(self, all_grads):
        self.batch_sizes = jnp.array([c.batch_size * c.epochs for c in self.network.clients])

    def scale(self, all_grads):
        idx = np.random.choice(self.K, size=int(self.C * self.K), replace=False)
        return jax.vmap(lambda b: b / self.batch_sizes[idx].sum())(self.batch_sizes[idx])