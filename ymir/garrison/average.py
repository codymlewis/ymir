r"""
Basic federated averaging proposed in `https://arxiv.org/abs/1602.05629 <https://arxiv.org/abs/1602.05629>`_
this simply scales received gradients, $\Delta$, by $\frac{1}{|\Delta|}$
"""

import numpy as np

import ymir.path

from . import captain


class Captain(captain.Captain):

    def __init__(self, params, network, rng=np.random.default_rng()):
        super().__init__(params, network, rng)

    def update(self, all_grads):
        """Update the stored batch sizes ($n_i$)."""
        pass

    def step(self):
        all_grads, _, all_losses = self.network(self.params, self.rng)

        # Captain side aggregation scaling
        self.update(all_grads)
        all_grads = [ymir.path.weights.scale(g, 1 / len(all_grads)) for g in all_grads]

        # Captain side update
        self.params = ymir.path.weights.sub(self.params, ymir.path.weights.add(*all_grads))
        return np.mean(all_losses)
