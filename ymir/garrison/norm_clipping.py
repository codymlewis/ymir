"""
Static norm clipping aggregator proposed in `https://arxiv.org/abs/1911.07963 <https://arxiv.org/abs/1911.07963>`_,
scales down any updates that sit out side of the $l_2$ sphere of radius $M$.
"""

import numpy as np

import ymir.path

from . import captain


class Captain(captain.Captain):

    def __init__(self, params, network, rng=np.random.default_rng(), M=1.0):
        """
        Construct the norm clipping aggregator.

        Optional arguments:
        - M: the radius of the $l_2$ sphere to scale according to.
        """
        super().__init__(params, network, rng)
        self.M = M

    def update(self, all_grads):
        pass

    def step(self):
        # Client side updates
        all_weights, _, all_losses = self.network(self.params, self.rng)
        # Captain side update
        Ws = [ymir.path.weights.ravel(w) for w in all_weights]
        all_weights = [
            ymir.path.weights.scale(w, 1 / max(1, np.linalg.norm(s, ord=2))) for w, s in zip(all_weights, Ws)
        ]
        self.params = ymir.path.weights.sub(self.params, ymir.path.weights.add(*all_weights))
        return np.mean(all_losses)
