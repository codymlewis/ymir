"""
Co-ordinate wise median algorithm proposed in
`http://proceedings.mlr.press/v80/yin18a/yin18a.pdf <http://proceedings.mlr.press/v80/yin18a/yin18a.pdf>`_
"""

import numpy as np

import ymir.path

from . import captain


class Captain(captain.Captain):

    def __init__(self, model, network, rng=np.random.default_rng()):
        r"""
        Construct the Median captain.
        """
        super().__init__(model, network, rng)
        self.unraveller = ymir.path.weights.unraveller(model.get_weights())

    def step(self):
        # Client side updates
        all_losses, all_grads, _ = self.network(self.model.get_weights(), self.rng)
        # Captain side update
        Ws = np.array([ymir.path.weights.ravel(w) for w in all_grads])
        median_weights = ymir.path.weights.unravel(np.median(Ws, axis=0), self.unraveller)
        self.model.optimizer.apply_gradients(zip(median_weights, self.model.weights))
        return np.mean(all_losses)
