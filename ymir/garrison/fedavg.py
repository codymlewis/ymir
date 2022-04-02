"""
Data respective federated averaging proposed in `https://arxiv.org/abs/1602.05629 <https://arxiv.org/abs/1602.05629>`_
this simply scales received gradients by the number of data they trained on divided by the total number of data,
$\\frac{n_i}{\sum_{i \in \mathcal{U}} n_i}$.
"""

import numpy as np

import ymir.path

from . import captain


class Captain(captain.Captain):

    def __init__(self, model, network, rng=np.random.default_rng()):
        super().__init__(model, network, rng)
        self.batch_sizes = np.array([c.batch_size * c.epochs for c in network.clients])

    def update(self, all_grads):
        pass

    def step(self):
        all_losses, all_grads, all_data = self.network(self.model.get_weights(), self.rng)
        # Captain-side aggregation scaling
        self.update(all_grads)
        N = sum(all_data)
        alpha = [n / N for n in all_data]
        all_grads = [ymir.path.weights.scale(g, a) for g, a in zip(all_grads, alpha)]
        # Captain-side update
        self.model.optimizer.apply_gradients(zip(ymir.path.weights.add(*all_grads), self.model.weights))
        return np.mean(all_losses)