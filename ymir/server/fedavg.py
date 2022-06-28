import jax
import jax.numpy as jnp

from ymir.utils import functions

from . import server


class Server(server.Server):

    def __init__(self, network, params, **kwargs):
        super().__init__(network, params, **kwargs)

    def step(self):
        all_params, all_loss, all_data = self.network(self.params)
        batch_sizes = jnp.array(all_data)
        self.update(functions.scale_sum(all_params, batch_sizes / batch_sizes.sum()))
        return jnp.mean(all_loss)
