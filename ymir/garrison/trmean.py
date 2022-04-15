"""
Trimmed mean algorithm proposed in
`http://proceedings.mlr.press/v80/yin18a/yin18a.pdf <http://proceedings.mlr.press/v80/yin18a/yin18a.pdf>`_
"""

import numpy as np

import ymir.path

from . import captain


class Captain(captain.Captain):

    def __init__(self, model, network, rng=np.random.default_rng(), beta=0.1):
        r"""
        Construct the Trimmed mean captain.
        
        Parameters:
        - beta: the beta parameter for the trimmed mean algorithm, states the half of the percentage of the client's updates to be removed
        """
        super().__init__(model, network, rng)
        self.unraveller = ymir.path.weights.unraveller(model.get_weights())
        self.beta = beta

    def step(self):
        # Client side updates
        all_losses, all_grads, _ = self.network(self.model.get_weights(), self.rng)
        # Captain side update
        Ws = np.array([ymir.path.weights.ravel(w) for w in all_grads])
        n_clients = Ws.shape[0]
        n_Ws_use = round(self.beta * n_clients)
        update_grad = np.sort(Ws, axis=0)[n_Ws_use:n_clients - n_Ws_use].sum(axis=0)
        gradients = ymir.path.weights.unravel((1 / ((1 - 2 * self.beta) * n_clients)) * update_grad, self.unraveller)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.weights))
        return np.mean(all_losses)
