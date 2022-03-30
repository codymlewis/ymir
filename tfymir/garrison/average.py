r"""
Basic federated averaging proposed in `https://arxiv.org/abs/1602.05629 <https://arxiv.org/abs/1602.05629>`_
this simply scales received gradients, $\Delta$, by $\frac{1}{|\Delta|}$
"""

import jax.numpy as jnp
import numpy as np

from . import captain


class Captain(captain.ScaleCaptain):

    def __init__(self, params, opt, opt_state, network, rng=np.random.default_rng()):
        super().__init__(params, opt, opt_state, network, rng)

    def update(self, all_grads):
        """Update the stored batch sizes ($n_i$)."""
        pass

    def scale(self, all_grads):
        return jnp.repeat(1 / len(all_grads), len(all_grads))
