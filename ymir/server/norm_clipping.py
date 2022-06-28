import jax
import jax.numpy as jnp

from . import server


class Server(server.Server):

    def __init__(self, network, params, **kwargs):
        super().__init__(network, params, **kwargs)

    def step(self):
        all_params, all_loss, _ = self.network(self.params)
        self.update(norm_clip(all_params))
        return jnp.mean(all_loss)


@jax.jit
def norm_clip(all_params):
    return jnp.einsum('Cw,C -> w', all_params, jnp.linalg.norm(all_params, axis=1, ord=2))
