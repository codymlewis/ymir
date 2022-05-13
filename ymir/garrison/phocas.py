"""
The Phocas algorithm proposed in `https://arxiv.org/abs/1805.09682 <https://arxiv.org/abs/1805.09682>`_
it is designed to provide robustness to generalized Byzantine attacks.
"""

import numpy as np

import ymir.path

from . import captain


class Captain(captain.Captain):

    def __init__(self, model, network, rng=np.random.default_rng(), beta=0.1):
        r"""
        Construct the Phocas captain.
        
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
        trmean = np.sort(Ws, axis=0)[n_Ws_use:n_clients - n_Ws_use].mean(axis=0)
        update_grad = np.take_along_axis(Ws, np.argsort(abs(Ws - trmean), axis=0),
                                         axis=0)[:n_clients - n_Ws_use].mean(axis=0)
        gradients = ymir.path.weights.unravel(update_grad, self.unraveller)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.weights))
        return np.mean(all_losses)
