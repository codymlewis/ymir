"""
The CONTRA algorithm proposed in `https://www.ittc.ku.edu/~bluo/pubs/Awan2021ESORICS.pdf <https://www.ittc.ku.edu/~bluo/pubs/Awan2021ESORICS.pdf>`_
it is designed to provide robustness to poisoning adversaries within many statistically heterogenous environments.
"""

import numpy as np
import sklearn.metrics.pairwise as smp

import ymir.path

from . import captain


class Captain(captain.Captain):

    def __init__(self, params, network, rng=np.random.default_rng(), C=0.1, k=10, delta=0.1, t=0.5):
        """
        Construct the CONTRA captain.

        Optional arguments:

        - C: Percentage of collaborators to be selected for each update.
        - k: Number of expected adversarial collaborators.
        - delta: Amount the increase/decrease the reputation (selection likelyhood) by.
        - t: Threshold for choosing when to increase the reputation.
        """
        super().__init__(params, network, rng)
        self.histories = np.zeros((len(network), len(ymir.path.weights.ravel(params))))
        self.C = C
        self.k = round(k * C)
        self.lamb = C * (1 - C)
        self.delta = delta
        self.t = t
        self.reps = np.ones(len(network))
        self.J = round(self.C * len(network))

    def update(self, all_grads):
        """Update the stored collaborator histories, that is, perform $H_{i, t + 1} \gets H_{i, t} + \Delta_{i, t + 1} : \\forall i \in \mathcal{U}$"""
        self.histories += np.array([ymir.path.weights.ravel(g) for g in all_grads])

    def scale(self, all_grads):
        n_clients = self.histories.shape[0]
        p = self.C + self.lamb * self.reps
        p[p <= 0] = 0
        p = p / p.sum()
        idx = np.random.choice(n_clients, size=self.J, p=p)
        L = idx.shape[0]
        cs = abs(smp.cosine_similarity(self.histories[idx])) - np.eye(L)
        cs[cs < 0] = 0
        taus = (-np.partition(-cs, self.k - 1, axis=1)[:, :self.k]).mean(axis=1)
        self.reps[idx] = np.where(taus > self.t, self.reps[idx] + self.delta, self.reps[idx] - self.delta)
        cs = cs * np.minimum(1, taus[:, None] / taus)
        taus = (-np.partition(-cs, self.k - 1, axis=1)[:, :self.k]).mean(axis=1)
        lr = np.zeros(n_clients)
        lr[idx] = 1 - taus
        self.reps[idx] = self.reps[idx] / self.reps[idx].max()
        lr[idx] = lr[idx] / lr[idx].max()
        lr[(lr == 1)] = .99  # eliminate division by zero in logit
        idx = idx[(lr[idx] > 0)]  # prevent inclusion of negatives in logit
        lr[idx] = np.log(lr[idx] / (1 - lr[idx])) + 0.5
        lr[(np.isinf(lr) + lr > 1)] = 1
        lr[(lr < 0)] = 0
        return lr

    def step(self):
        all_grads, _, all_losses = self.network(self.params, self.rng)

        # Captain side aggregation scaling
        self.update(all_grads)
        alpha = self.scale(all_grads)
        all_grads = [ymir.path.weights.scale(g, a) for g, a in zip(all_grads, alpha)]

        # Captain side update
        self.params = ymir.path.weights.add(self.params, ymir.path.weights.add(*all_grads))
        return np.mean(all_losses)
