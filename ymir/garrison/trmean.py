"""
Trimmed mean algorithm proposed in
`http://proceedings.mlr.press/v80/yin18a/yin18a.pdf <http://proceedings.mlr.press/v80/yin18a/yin18a.pdf>`_
"""

import numpy as np

import ymir.path

from . import captain


class Captain(captain.Captain):

    def __init__(self, params, network, rng=np.random.default_rng(), beta=0.1):
        r"""
        Construct the Trimmed mean captain.
        
        Parameters:
        - beta: the beta parameter for the trimmed mean algorithm, states the half of the percentage of the client's updates to be removed
        """
        super().__init__(params, network, rng)
        self.skeleton = ymir.path.weights.skeleton(params)
        self.beta = beta

    def update(self, all_weights):
        Ws = np.array([ymir.path.weights.ravel(w) for w in all_weights])
        n_clients = Ws.shape[0]
        n_Ws_use = round(self.beta * n_clients)
        update_weight = np.sort(Ws, axis=0)[n_Ws_use:n_clients - n_Ws_use].sum(axis=0)
        return ymir.path.weights.unravel((1 / ((1 - 2 * self.beta) * n_clients)) * update_weight, self.skeleton)

    def step(self):
        # Client side updates
        all_weights, _, all_losses = self.network(self.params, self.rng)
        # Captain side update
        self.params = ymir.path.weights.add(self.params, self.update(all_weights))
        return np.mean(all_losses)
