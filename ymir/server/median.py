import einops
import jax
import jax.numpy as jnp

from . import server


class Server(server.Server):

    def __init__(self, network, params, **kwargs):
        super().__init__(network, params, **kwargs)

    def step(self):
        all_params, all_loss, _ = self.network(self.params)
        self.update(median(all_params))
        return jnp.mean(all_loss)


@jax.jit
def median(all_params):
    return einops.reduce(all_params, 'C w -> w', jnp.median)
