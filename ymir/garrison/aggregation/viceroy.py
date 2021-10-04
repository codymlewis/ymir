from dataclasses import dataclass
import sys

import numpy as np
import sklearn.metrics.pairwise as smp
import jax
import jax.numpy as jnp
import optax

import jaxlib


"""
The viceroy algorithm

Call order: server init -> update -> scale
"""


@dataclass
class Server:
    histories: jaxlib.xla_extension.DeviceArray
    reps: jaxlib.xla_extension.DeviceArray

    def __init__(self, n_clients, params):
        self.histories = jnp.zeros((n_clients, jax.flatten_util.ravel_pytree(params)[0].shape[0]))
        self.reps = jnp.array([0.01 for _ in range(n_clients)])



@jax.jit
def update(histories, rep, all_grads, omega=0.85, eta=0.25, epsilon=-0.0001):
    """Performed after a scale"""
    X = jnp.array([jax.flatten_util.ravel_pytree(g)[0] for g in all_grads])
    r = abs(optax.cosine_similarity(histories + sys.float_info.epsilon, X))
    idx = rep >= epsilon
    rep = jnp.where(
        idx,
        jnp.clip(omega * rep + eta * jnp.tanh(2 * r - 1), -1, 1),
        omega**r * rep
    )
    histories = omega * histories + (X.T * (rep >= 0)).T
    rep = jnp.where(idx * ((r + (rep / 2)) < 0.5), -1, rep)
    return histories, rep


def scale(histories, reps):
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
    wv[idx] = (np.log(wv[idx] / (1 - wv[idx])) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0
    return wv * np.maximum(reps, 0)