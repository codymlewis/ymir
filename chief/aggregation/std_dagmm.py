import sys
from dataclasses import dataclass
from haiku._src.basic import Linear

import jax
from jax._src.numpy.lax_numpy import expand_dims
import jax.flatten_util
import jax.numpy as jnp
import haiku as hk
import optax
import jaxlib

from sklearn import mixture

class DA(hk.Module):
    def __init__(self, in_len, n_gmm=2, latent_dim=4, name=None):
        super().__init__(name=name)
        self.encoder = hk.Sequential([
            hk.Linear(60), jax.nn.relu,
            hk.Linear(30), jax.nn.relu,
            hk.Linear(10), jax.nn.relu,
            hk.Linear(1)
        ])
        self.decoder = hk.Sequential([
            hk.Linear(10), jax.nn.tanh,
            hk.Linear(30), jax.nn.tanh,
            hk.Linear(60), jax.nn.tanh,
            hk.Linear(in_len)
        ])

    def __call__(self, X):
        enc = self.encoder(X)
        return enc, self.decoder(enc)


def loss(net):
    @jax.jit
    def _apply(params, x):
        _, z = net.apply(params, x)
        return jnp.mean(optax.l2_loss(z, x))
    return _apply

def da_update(opt, loss):
    @jax.jit
    def _apply(params, opt_state, batch):
        grads = jax.grad(loss)(params, batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    return _apply


@jax.jit
def relative_euclidean_distance(a, b):
    return jnp.linalg.norm(a - b, ord=2) / jnp.clip(jnp.linalg.norm(a, ord=2), a_min=1e-10)

class Server:
    def __init__(self, batch_sizes, x):
        self.batch_sizes = batch_sizes
        x = jnp.array([jax.flatten_util.ravel_pytree(x)[0]])
        self.da = hk.without_apply_rng(hk.transform(lambda x: DA(x[0].shape[0])(x)))
        rng = jax.random.PRNGKey(42)
        self.params = self.da.init(rng, x)
        opt = optax.adamw(0.001, weight_decay=0.0001)
        self.opt_state = opt.init(self.params)
        self.update = da_update(opt, loss(self.da))

        self.gmm = mixture.GaussianMixture(4, random_state=0, warm_start=True)


def update(server, grads):
    grads = jnp.array([jax.flatten_util.ravel_pytree(g)[0].tolist() for g in grads])
    params, opt_state = server.update(server.params, server.opt_state, grads)
    enc, dec = server.da.apply(params, grads)
    z = jnp.array([[
        jnp.squeeze(e),
        relative_euclidean_distance(x, d),
        optax.cosine_similarity(x, d),
        jnp.std(x)
    ] for x, e, d in zip(grads, enc, dec)])
    server.gmm = server.gmm.fit(z)
    return params, opt_state


def predict(params, net, gmm, X):
    enc, dec = net.apply(params, X)
    z = jnp.array([[
        jnp.squeeze(e),
        relative_euclidean_distance(x, d),
        optax.cosine_similarity(x, d),
        jnp.std(x)
    ] for x, e, d in zip(X, enc, dec)])
    return gmm.score_samples(z)

def scale(batch_sizes, grads, server):
    grads = jnp.array([jax.flatten_util.ravel_pytree(g)[0].tolist() for g in grads])
    energies = predict(server.params, server.da, server.gmm, grads)
    std = jnp.std(energies)
    avg = jnp.mean(energies)
    mask = jnp.where((energies >= avg - std) * (energies <= avg + std), 1, 0)
    total_dc = jnp.sum(batch_sizes * mask)
    return (batch_sizes / total_dc) * mask