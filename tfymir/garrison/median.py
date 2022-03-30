"""
Co-ordinate wise median algorithm proposed in
`http://proceedings.mlr.press/v80/yin18a/yin18a.pdf <http://proceedings.mlr.press/v80/yin18a/yin18a.pdf>`_
"""

import jax
import jax.flatten_util
import numpy as np

from . import captain


class Captain(captain.AggregateCaptain):

    def __init__(self, params, opt, opt_state, network, rng=np.random.default_rng()):
        r"""
        Construct the Median captain.
        """
        super().__init__(params, opt, opt_state, network, rng)
        self.G_unraveller = jax.flatten_util.ravel_pytree(params)[1]

    def update(self, all_weights):
        Ws = np.array([jax.flatten_util.ravel_pytree(w)[0] for w in all_weights])
        return self.G_unraveller(np.median(Ws, axis=0))

    def step(self):
        # Client side updates
        all_weights = self.network(self.params, self.rng)

        # Captain side update
        self.params, self.opt_state = self.update_params(self.params, self.opt_state, self.update(all_weights))
