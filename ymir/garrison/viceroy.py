"""
Viceroy algorithm for mitigating the impact of the on-off toggling attack.
"""

import sys

import numpy as np
import sklearn.metrics.pairwise as smp

import ymir.path

from . import captain


class Captain(captain.Captain):

    def __init__(self, model, network, rng=np.random.default_rng(), tau_0=56, tau_1=5):
        r"""
        Construct the Viceroy captain.

        Optional arguments:
        - tau_0: amount of rounds for the reputation to decay to 0 ($\tau_0$).
        - tau_1: amount of rounds for the reputation to build to 1 ($\tau_1$).
        """
        super().__init__(model, network, rng)
        self.histories = np.zeros((len(network), len(ymir.path.weights.ravel(model.get_weights()))))
        self.reps = np.array([1.0 for _ in range(len(network))])
        self.round = 1
        self.omega = (abs(sys.float_info.epsilon))**(1 / tau_0)
        self.eta = 1 / tau_1

    def update(self, all_grads):
        if self.round > 1:
            history_scale = fedscale(self.histories)
            current_scale = fedscale(np.array([ymir.path.weights.ravel(g) for g in all_grads]))
            self.reps = np.clip(self.reps + ((1 - 2 * abs(history_scale - current_scale)) / 2) * self.eta, 0, 1)
        self.round += 1
        self.histories = np.array(
            [self.omega * h + ymir.path.weights.ravel(g) for h, g in zip(self.histories, all_grads)]
        )

    def scale(self, all_grads):
        return (self.reps * fedscale(self.histories)
                ) + ((1 - self.reps) * fedscale(np.array([ymir.path.weights.ravel(g) for g in all_grads])))

    def step(self):
        all_losses, all_grads, _ = self.network(self.model.get_weights(), self.rng)
        # Captain-side aggregation scaling
        self.update(all_grads)
        all_grads = [ymir.path.weights.scale(g, a) for g, a in zip(all_grads, self.scale(all_grads))]
        # Captain-side update
        self.model.optimizer.apply_gradients(zip(ymir.path.weights.add(*all_grads), self.model.weights))
        return np.mean(all_losses)


def fedscale(X):
    """A modified FoolsGold algorithm for scaling the gradients/histories."""
    n_clients = X.shape[0]
    cs = smp.cosine_similarity(X) - np.eye(n_clients)
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
    wv[idx] = (np.log(wv[idx] / (1 - wv[idx])) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0
    return wv
