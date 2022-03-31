"""
Co-ordinate wise median algorithm proposed in
`http://proceedings.mlr.press/v80/yin18a/yin18a.pdf <http://proceedings.mlr.press/v80/yin18a/yin18a.pdf>`_
"""

import numpy as np

import ymir.path

from . import captain


class Captain(captain.Captain):

    def __init__(self, params, network, rng=np.random.default_rng()):
        r"""
        Construct the Median captain.
        """
        super().__init__(params, network, rng)
        self.skeleton = ymir.path.weights.skeleton(params)

    def update(self, all_weights):
        pass

    def step(self):
        # Client side updates
        all_weights, _, all_losses = self.network(self.params, self.rng)
        # Captain side update
        Ws = np.array([ymir.path.weights.ravel(w) for w in all_weights])
        median_weights = ymir.path.weights.unravel(np.median(Ws, axis=0), self.skeleton)
        self.params = ymir.path.weights.add(self.params, median_weights)
        return np.mean(all_losses)
