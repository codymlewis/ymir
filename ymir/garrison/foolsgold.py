"""
The FoolsGold algorithm proposed in `https://arxiv.org/abs/1808.04866 <https://arxiv.org/abs/1808.04866>`_
"""

import numpy as np
import sklearn.metrics.pairwise as smp
import jax
import jax.numpy as jnp

from . import captain


class Captain(captain.ScaleCaptain):
    def __init__(self, params, opt, opt_state, network, rng=np.random.default_rng(), kappa=1.0):
        super().__init__(params, opt, opt_state, network, rng)
        self.histories = jnp.zeros((len(network), jax.flatten_util.ravel_pytree(params)[0].shape[0]))
        self.kappa = kappa

    def update(self, all_grads):
        self.histories = update(self.histories, all_grads)

    def scale(self, all_grads):
        """Adapted from https://github.com/DistributedML/FoolsGold"""
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

@jax.jit
def update(histories, all_grads):
    return jnp.array([h + jax.flatten_util.ravel_pytree(g)[0] for h, g in zip(histories, all_grads)])