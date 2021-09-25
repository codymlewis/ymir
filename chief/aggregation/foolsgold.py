from dataclasses import dataclass

import numpy as np
import sklearn.metrics.pairwise as smp
import jax
import jax.numpy as jnp

import jaxlib


@dataclass
class Server:
    histories: jaxlib.xla_extension.DeviceArray
    kappa: float

    def __init__(self, n_clients, params, kappa):
        self.histories = jnp.zeros((n_clients, jax.flatten_util.ravel_pytree(params)[0].shape[0]))
        self.kappa = kappa


@jax.jit
def update(histories, all_grads):
    return jnp.array([h + jax.flatten_util.ravel_pytree(g)[0] for h, g in zip(histories, all_grads)])

def scale(histories, kappa):
    """Adapted from https://github.com/DistributedML/FoolsGold"""
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