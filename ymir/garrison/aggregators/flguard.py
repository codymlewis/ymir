from dataclasses import dataclass

import numpy as np
import sklearn.metrics.pairwise as smp
import jax
import jax.flatten_util
import jax.numpy as jnp
import hdbscan

from . import server


"""
The FLGuard algorithm proposed in https://arxiv.org/abs/2101.02281
"""

class Server(server.AggServer):
    def __init__(self, params, opt, opt_state, network, rng, lamb=0.001):
        super().__init__(params, opt, opt_state, network, rng)
        self.G_unraveller = jax.flatten_util.ravel_pytree(params)[1]
        self.lamb = lamb  # 0.001 is good for image classification, 0.01 for IDS (according to the paper)

    def update(self, all_weights):
        G = np.array(jax.flatten_util.ravel_pytree(self.params)[0])
        Ws = np.array([jax.flatten_util.ravel_pytree(w)[0] for w in all_weights])
        n_clients = Ws.shape[0]
        cs = smp.cosine_distances(Ws).astype(np.double)
        clusters = hdbscan.HDBSCAN(min_cluster_size=n_clients // 2 + 1, metric='precomputed', allow_single_cluster=True).fit_predict(cs)
        bs = np.arange(len(clusters))[clusters == np.argmax(np.bincount(clusters[clusters != -1]))]
        es = np.linalg.norm(G - Ws, axis=1)  # Euclidean distance between G and each Ws
        S = np.median(es)
        Ws[bs] = (Ws[bs].T * np.minimum(1, S / es[bs])).T
        G = Ws[bs].mean(axis=0)
        sigma = self.lamb / S
        G = G + self.rng.normal(0, sigma, G.shape)
        return self.G_unraveller(G)

    def scale(self, all_grads):
        pass
        
    
    def step(self):
        # Client side updates
        all_weights = self.network(self.params, self.rng, return_weights=True)

        # Server side update
        self.params = self.update(all_weights)
        # return alpha, all_grads