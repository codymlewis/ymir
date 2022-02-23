"""
Trimmed mean algorithm proposed in
`http://proceedings.mlr.press/v80/yin18a/yin18a.pdf <http://proceedings.mlr.press/v80/yin18a/yin18a.pdf>`_
"""

import jax
import jax.flatten_util
import numpy as np

from . import captain


class Captain(captain.AggregateCaptain):

    def __init__(self, params, opt, opt_state, network, rng=np.random.default_rng(), beta=0.1):
        r"""
        Construct the Trimmed mean captain.
        
        Parameters:
        - beta: the beta parameter for the trimmed mean algorithm, states the half of the percentage of the client's updates to be removed
        """
        super().__init__(params, opt, opt_state, network, rng)
        self.G_unraveller = jax.flatten_util.ravel_pytree(params)[1]
        self.beta = beta

    def update(self, all_weights):
        Ws = np.array([jax.flatten_util.ravel_pytree(w)[0] for w in all_weights])
        n_clients = Ws.shape[0]
        n_Ws_use = round(self.beta * n_clients)
        update_weight = np.sort(Ws, axis=0)[n_Ws_use:n_clients - n_Ws_use].sum(axis=0)
        return self.G_unraveller((1 / ((1 - 2 * self.beta) * n_clients)) * update_weight)

    def step(self):
        # Client side updates
        all_weights = self.network(self.params, self.rng)

        # Captain side update
        self.params, self.opt_state = self.update_params(self.params, self.opt_state, self.update(all_weights))
