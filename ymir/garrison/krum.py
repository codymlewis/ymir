"""
The multi-Krum algorithm proposed in `https://papers.nips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html <https://papers.nips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html>`_
it is designed to be robust to Byzantine faults with i.i.d. environments.
"""

import numpy as np

import ymir.path

from . import captain


class Captain(captain.Captain):

    def __init__(self, model, network, rng=np.random.default_rng(), clip=3):
        """
        Construct the Krum captain.

        Optional arguments:
        - clip: the number of expected faults in each round.
        """
        super().__init__(model, network, rng)
        self.clip = clip

    def scale(self, all_grads):
        n = len(all_grads)
        X = np.array([ymir.path.weights.ravel(g) for g in all_grads])
        scores = np.zeros(n)
        distances = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None] - 2 * np.dot(X, X.T)
        for i in range(len(X)):
            scores[i] = np.sum(np.sort(distances[i])[1:((n - self.clip) - 1)])
        idx = np.argpartition(scores, n - self.clip)[:(n - self.clip)]
        alpha = np.zeros(n)
        alpha[idx] = 1
        return alpha

    def step(self):
        # Client side updates
        all_losses, all_grads, _ = self.network(self.model.get_weights(), self.rng)
        # Captain side update
        all_grads = [ymir.path.weights.scale(g, a) for g, a in zip(all_grads, self.scale(all_grads))]
        self.model.optimizer.apply_gradients(zip(ymir.path.weights.add(*all_grads), self.model.weights))
        return np.mean(all_losses)
