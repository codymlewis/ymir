"""
The FoolsGold algorithm proposed in `https://arxiv.org/abs/1808.04866 <https://arxiv.org/abs/1808.04866>`_
it is designed to provide robustness to (Sybil) adversarial attacks within non-i.i.d. environments.
"""

import numpy as np
import sklearn.metrics.pairwise as smp

import ymir.path

from . import captain


class Captain(captain.Captain):

    def __init__(self, params, network, rng=np.random.default_rng(), kappa=1.0):
        """
        Construct the FoolsGold captain.

        Optional arguments:
        - kappa: value stating the distribution of classes across clients.
        """
        super().__init__(params, network, rng)
        self.histories = np.zeros((len(network), len(ymir.path.weights.ravel(params))))
        self.kappa = kappa

    def update(self, all_grads):
        self.histories += np.array([ymir.path.weights.ravel(g) for g in all_grads])

    def scale(self, all_grads):
        """
        Scale the gradients according to the FoolsGold algorithm.

        Code adapted from `https://github.com/DistributedML/FoolsGold <https://github.com/DistributedML/FoolsGold>`_.
        """
        n_clients = self.histories.shape[0]
        cs = smp.cosine_similarity(self.histories) - np.eye(n_clients)
        maxcs = np.max(cs, axis=1)
        # pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        wv = 1 - (np.max(cs, axis=1))
        wv[wv > 1] = 1
        wv[wv < 0] = 0
        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99
        # Logit function
        idx = wv != 0
        wv[idx] = self.kappa * (np.log(wv[idx] / (1 - wv[idx])) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0
        return wv

    def step(self):
        # Client side updates
        all_grads, _, all_losses = self.network(self.params, self.rng)
        # Captain side update
        self.update(all_grads)
        alpha = self.scale(all_grads)
        all_grads = [ymir.path.weights.scale(g, a) for g, a in zip(all_grads, alpha)]
        self.params = ymir.path.weights.sub(self.params, ymir.path.weights.add(*all_grads))
        return np.mean(all_losses)
