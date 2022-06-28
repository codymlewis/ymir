"""
The CONTRA algorithm proposed in `https://www.ittc.ku.edu/~bluo/pubs/Awan2021ESORICS.pdf <https://www.ittc.ku.edu/~bluo/pubs/Awan2021ESORICS.pdf>`_
it is designed to provide robustness to poisoning adversaries within many statistically heterogenous environments.
"""

import jax.numpy as jnp
import numpy as np
import sklearn.metrics.pairwise as smp

from ymir.utils import functions

from . import server


class Server(server.Server):

    def __init__(self, network, params, C=0.1, k=10, delta=0.1, t=0.5, **kwargs):
        """
        Construct the FoolsGold server.
        Optional arguments:
        - kappa: value stating the distribution of classes across clients.
        """
        super().__init__(network, params, **kwargs)
        self.histories = np.zeros((len(network), self.params_len))
        self.C = C
        self.k = round(k * C)
        self.lamb = C * (1 - C)
        self.delta = delta
        self.t = t
        self.reps = np.ones(len(network))
        self.J = round(self.C * len(network))

    def step(self):
        all_params, all_loss, _ = self.network(self.params)
        self.histories += np.array(all_params)
        self.update(functions.scale_sum(all_params, self.contra()))
        return jnp.mean(all_loss)

    def contra(self):
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
