import einops
import hdbscan
import jax.numpy as jnp
import numpy as np
import sklearn.metrics.pairwise as smp

from ymir.utils import functions

from . import server


class Server(server.Server):

    def __init__(self, network, params, eps=3705, delta=1, **kwargs):
        super().__init__(network, params, **kwargs)
        self.lamb = (1 / eps) * jnp.sqrt(2 * jnp.log(1.25 / delta))

    def step(self):
        all_weights, all_loss, _ = self.network(self.params, return_weights=True)
        self.params = self.unraveller(
            flame(np.array(functions.ravel(self.params)), np.array(all_weights), self.lamb, self.rng)
        )
        return jnp.mean(all_loss)


def flame(G, Ws, lamb, rng):
    n_clients = Ws.shape[0]
    cs = smp.cosine_distances(Ws).astype(np.double)
    clusters = hdbscan.HDBSCAN(min_cluster_size=n_clients // 2 + 1, metric='precomputed',
                               allow_single_cluster=True).fit_predict(cs)
    bs = np.arange(len(clusters))[clusters == np.argmax(np.bincount(clusters[clusters != -1]))]
    es = np.linalg.norm(G - Ws, axis=1)  # Euclidean distance between G and each Ws
    S = np.median(es)
    Ws[bs] = G + np.einsum('Bw,B -> Bw', Ws[bs] - G, np.minimum(1, S / es[bs]))
    G = einops.reduce(Ws[bs], 'B w -> w', np.mean)
    sigma = lamb * S
    G = G + rng.normal(0, sigma, G.shape)
    return G
