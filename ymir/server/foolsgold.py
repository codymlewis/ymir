"""
The FoolsGold algorithm proposed in `https://arxiv.org/abs/1808.04866 <https://arxiv.org/abs/1808.04866>`_
it is designed to provide robustness to (Sybil) adversarial attacks within non-i.i.d. environments.
"""

import jax.numpy as jnp
import numpy as np
import sklearn.metrics.pairwise as smp

from ymir.utils import functions

from . import server


class Server(server.Server):

    def __init__(self, network, params, kappa=1.0, **kwargs):
        """
        Construct the FoolsGold server.
        Optional arguments:
        - kappa: value stating the distribution of classes across clients.
        """
        super().__init__(network, params, **kwargs)
        self.histories = np.zeros((len(network), self.params_len))
        self.kappa = kappa

    def step(self):
        all_params, all_loss, _ = self.network(self.params)
        self.histories += np.array(all_params)
        self.update(functions.scale_sum(all_params, foolsgold(self.histories, self.kappa)))
        return jnp.mean(all_loss)


def foolsgold(histories, kappa):
    """
    Scale the gradients according to the FoolsGold algorithm.
    Code adapted from `https://github.com/DistributedML/FoolsGold <https://github.com/DistributedML/FoolsGold>`_.
    """
    n_clients = histories.shape[0]
    cs = smp.cosine_similarity(histories) - np.eye(n_clients)
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
    wv[idx] = kappa * (np.log(wv[idx] / (1 - wv[idx])) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0
    return wv
