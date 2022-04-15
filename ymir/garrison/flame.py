"""
The FLAME algorithm proposed in `https://arxiv.org/abs/2101.02281 <https://arxiv.org/abs/2101.02281>`_
it is designed to provide robustness against adversaries, inclusive of multiple attacks and statistical
heterogeneity environments.
"""

import hdbscan
import numpy as np
import sklearn.metrics.pairwise as smp

import ymir.path

from . import captain


class Captain(captain.Captain):

    def __init__(self, params, network, rng=np.random.default_rng(), eps=3705, delta=1):
        r"""
        Construct the FLAME captain.

        Optional arguments:
        - eps: the epsilon parameter for the FLAME algorithm, respective to ($\epsilon, \delta$)-DP
        - delta: the delta parameter for the FLAME algorithm, respective to ($\epsilon, \delta$)-DP
        """
        super().__init__(params, network, rng)
        self.unraveller = ymir.path.weights.unraveller(params)
        self.lamb = (1 / eps) * np.sqrt(2 * np.log(1.25 / delta))

    def update(self, all_weights):
        G = ymir.path.weights.ravel(self.params)
        Ws = np.array([ymir.path.weights.ravel(w) for w in all_weights])
        n_clients = Ws.shape[0]
        cs = smp.cosine_distances(Ws).astype(np.double)
        clusters = hdbscan.HDBSCAN(
            min_cluster_size=n_clients // 2 + 1, metric='precomputed', allow_single_cluster=True
        ).fit_predict(cs)
        bs = np.arange(len(clusters))[clusters == np.argmax(np.bincount(clusters[clusters != -1]))]
        es = np.linalg.norm(G - Ws, axis=1)  # Euclidean distance between G and each Ws
        S = np.median(es)
        Ws[bs] = G + ((Ws[bs] - G).T * np.minimum(1, S / es[bs])).T
        G = Ws[bs].mean(axis=0)
        sigma = self.lamb * S
        G = G + self.rng.normal(0, sigma, G.shape)
        return ymir.path.weights.unravel(G, self.unraveller)

    def step(self):
        # Client side updates
        all_weights, _, all_losses = self.network(self.params, self.rng, return_weights=True)

        # Captain side update
        self.params = self.update(all_weights)
        return np.mean(all_losses)
