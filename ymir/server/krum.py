"""
The multi-Krum algorithm proposed in `https://papers.nips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html <https://papers.nips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html>`_
it is designed to be robust to Byzantine faults with i.i.d. environments.
"""

import jax.numpy as jnp
import numpy as np

from ymir.utils import functions

from . import server


class Server(server.Server):

    def __init__(self, network, params, clip=3, **kwargs):
        """
        Construct the FoolsGold server.
        Optional arguments:
        - clip: the number of expected faults in each round.
        """
        super().__init__(network, params, **kwargs)
        self.clip = clip

    def step(self):
        all_params, all_loss, _ = self.network(self.params)
        self.update(functions.scale_sum(all_params, krum(all_params, len(all_params), self.clip)))
        return jnp.mean(all_loss)


def krum(X, n, clip):
    n = len(X)
    scores = np.zeros(n)
    distances = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None] - 2 * np.dot(X, X.T)
    for i in range(len(X)):
        scores[i] = np.sum(np.sort(distances[i])[1:((n - clip) - 1)])
    idx = np.argpartition(scores, n - clip)[:(n - clip)]
    alpha = np.zeros(n)
    alpha[idx] = 1
    return alpha
