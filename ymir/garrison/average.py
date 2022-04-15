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

    def step(self):
        all_losses, all_grads, _ = self.network(self.model.get_weights(), self.rng)
        all_grads = [ymir.path.weights.scale(g, 1 / len(all_grads)) for g in all_grads]
        self.model.optimizer.apply_gradients(zip(ymir.path.weights.add(*all_grads), self.model.weights))
        return np.mean(all_losses)
