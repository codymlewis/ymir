import jax
import jax.numpy as jnp

from . import server


"""
Basic federated averaging proposed in https://arxiv.org/abs/1602.05629
"""


class Server(server.AggServer):
    def __init__(self, params, network):
        self.batch_sizes = jnp.array([c.batch_size for c in network.clients])
    
    def update(self, all_grads):
        pass

    def scale(self, all_grads):
        return jax.vmap(lambda b: b / self.batch_sizes.sum())(self.batch_sizes)