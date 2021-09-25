from dataclasses import dataclass
import sys

import numpy as np
import sklearn.metrics.pairwise as smp
import jax
import jax.numpy as jnp
import optax

import jaxlib


@dataclass
class Server:
    histories: jaxlib.xla_extension.DeviceArray
    reps: jaxlib.xla_extension.DeviceArray

    def __init__(self, n_clients, params):
        self.histories = jnp.zeros((n_clients, jax.flatten_util.ravel_pytree(params)[0].shape[0]))
        self.reps = jnp.array([0.01 for _ in range(n_clients)])


@jax.jit
def init(all_grads):
    """Initialize the histories"""
    return jnp.array([jax.flatten_util.ravel_pytree(g)[0] for g in all_grads])

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

@jax.jit
def scale(grads, histories, reps, T):
    n = histories.shape[0]
    X = jnp.array([jax.flatten_util.ravel_pytree(g)[0] for g in grads])
    G = jnp.sum(X, axis=0) / n
    S = jnp.sum(histories, axis=0) / (T * n)
    alpha = jnp.exp(
        -jnp.where(T == 1, optax.cosine_similarity(X, G), (optax.cosine_similarity(X, S) - (optax.cosine_similarity(X, G) / 2)))**2
    ) * jnp.maximum(reps, 0)
    alpha = jnp.where(alpha > 0, alpha / jnp.sum(alpha), 0)
    return alpha