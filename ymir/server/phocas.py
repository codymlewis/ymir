"""
The Phocas algorithm proposed in `https://arxiv.org/abs/1805.09682 <https://arxiv.org/abs/1805.09682>`_
it is designed to provide robustness to generalized Byzantine attacks.
"""

from functools import partial

import einops
import jax
import jax.numpy as jnp
import numpy as np

from ymir.utils import functions

from . import server


class Server(server.Server):

    def __init__(self, network, params, beta=0.1, **kwargs):
        """
        Construct the FoolsGold server.
        Optional arguments:
        - beta: the beta parameter for the trimmed mean algorithm, states the half of the percentage of the client's updates to be removed
        """
        super().__init__(network, params, **kwargs)
        self.beta = round(beta * len(network))

    def step(self):
        all_params, all_loss, _ = self.network(self.params)
        self.update(phocas(all_params, self.beta))
        return jnp.mean(all_loss)


@partial(jax.jit, static_argnums=(1, ))
def phocas(Ws, beta):
    n_clients = Ws.shape[0]
    trmean = einops.reduce(jnp.sort(Ws, axis=0)[beta:n_clients - beta], 'C w -> w', jnp.mean)
    return einops.reduce(
        jnp.take_along_axis(Ws, jnp.argsort(abs(Ws - trmean), axis=0), axis=0)[:n_clients - beta], 'C w -> w', jnp.mean
    )
