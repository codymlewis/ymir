from functools import partial

import einops
import jax
import jax.numpy as jnp

from . import server


class Server(server.Server):

    def __init__(self, network, params, beta=0.1, **kwargs):
        super().__init__(network, params, **kwargs)
        self.beta = round(beta * len(network))

    def step(self):
        all_params, all_loss, _ = self.network(self.params)
        self.update(trmean(all_params, self.beta))
        return jnp.mean(all_loss)


@partial(jax.jit, static_argnums=(1, ))
def trmean(all_params, beta):
    n_clients = all_params.shape[0]
    return einops.reduce(jnp.sort(all_params, axis=0)[beta:n_clients - beta], 'C w -> w', jnp.mean)
