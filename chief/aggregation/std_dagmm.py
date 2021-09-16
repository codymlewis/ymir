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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class STDDAGMM(hk.Module):
    def __init__(self, in_len, n_gmm=2, latent_dim=4, name=None):
        super().__init__(name=name)
        self.encoder = hk.Sequential([
            # hk.Linear(in_len),
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
        self.n_gmm = n_gmm
        self.phi = jnp.zeros(n_gmm)
        self.mu = jnp.zeros((n_gmm, latent_dim))
        self.cov = jnp.zeros((n_gmm, latent_dim, latent_dim))
        self.opt = optax.adamw(0.001, weight_decay=0.0001)
    
    def estimate(self, x):
        x = hk.Linear(10)(x)
        x = jax.nn.tanh(x)
        x = hk.dropout(hk.next_rng_key(), 0.5, x)
        x = hk.Linear(self.n_gmm)(x)
        x = jnp.exp(x)
        return x / jnp.sum(x)

    def __call__(self, X):
        enc = self.encoder(X)
        dec = self.decoder(enc)
        z = jnp.array([[
            jnp.squeeze(e),
            relative_euclidean_distance(x, d),
            optax.cosine_similarity(x, d),
            jnp.std(x)
        ] for x, e, d in zip(X, enc, dec)])
        gamma = self.estimate(z)
        return enc, dec, z, gamma

    def compute_gmm_params(self, z, gamma):
        N = gamma.size
        sum_gamma = jnp.sum(gamma, axis=0)
        phi = sum_gamma / N
        self.phi = phi
        mu = jnp.sum(jnp.expand_dims(gamma, -1) * jnp.expand_dims(z, 1), axis=0) / jnp.expand_dims(sum_gamma, -1)
        self.mu = mu
        z_mu = jnp.expand_dims(z, 1) - jnp.expand_dims(mu, 0)
        z_mu_outer = jnp.expand_dims(z_mu, -1) * jnp.expand_dims(z_mu, -2)
        cov = jnp.sum(jnp.expand_dims(jnp.expand_dims(gamma, -1), -1) * z_mu_outer, axis=0) / jnp.expand_dims(jnp.expand_dims(sum_gamma, -1), -1)
        self.cov = cov
        return phi, mu, cov

    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = self.phi
        if mu is None:
            mu = self.mu
        if cov is None:
            cov = self.cov
        k, d, _ = cov.shape
        z_mu = jnp.expand_dims(z, 1) - jnp.expand_dims(mu, 0)
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            cov_k = cov[i] + (jnp.eye(d) * eps)
            pinv = jnp.linalg.pinv(cov_k)
            cov_inverse.append(pinv)
            eigvals = jnp.linalg.eigvals(cov_k * (2 * jnp.pi))
            determinant = jnp.prod(jnp.clip(eigvals, a_min=sys.float_info.epsilon, a_max=None))
            det_cov.append(determinant)
            cov_diag += jnp.sum((1 / cov_k) * jnp.eye(cov_k.shape[0]))
        cov_inverse = jnp.array(cov_inverse)
        det_cov = jnp.array(det_cov, dtype=jnp.float32)
        exp_term_tmp = -0.5 * jnp.sum(jnp.sum(jnp.expand_dims(z_mu, -1) * jnp.expand_dims(cov_inverse, 0), axis=-2) * z_mu, axis=-1)
        max_val = jnp.max(jnp.clip(exp_term_tmp, a_min=0), axis=1, keepdims=True)[0]
        exp_term = jnp.exp(exp_term_tmp - max_val)
        sample_energy = -max_val.squeeze() - jnp.log(
                jnp.sum(
                    jnp.expand_dims(phi, 0) * exp_term /
                    jnp.expand_dims(jnp.sqrt(det_cov) + eps, 0),
                    axis=1
                ) + eps
        )
        if size_average:
            sample_energy = jnp.mean(sample_energy)
        return sample_energy, cov_diag

    def predict(self, X):
        E = []
        for x in X:
            _, _, z, _ = self(x.unsqueeze(0))
            e, _ = self.compute_energy(z, size_average=False)
            E = torch.cat((E, e))
            E.append(e)
        return jnp.array(E)


def loss(net, netcgp, netce):
    # @jax.jit
    def _apply(params, rng, x, lambda_energy, lambda_cov_diag):
        _, x_hat, z, gamma = net.apply(params, rng, x)
        recon_error = jnp.mean((x - x_hat) ** 2)
        phi, mu, cov = netcgp.apply(params, z, gamma)
        sample_energy, cov_diag = netce.apply(params, z, phi, mu, cov, True)
        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag
        return loss
    return lambda p, r, x: _apply(p, r, x, 0.1, 0.005)

def relative_euclidean_distance(a, b):
    return jnp.linalg.norm(a - b, ord=2) / jnp.clip(jnp.linalg.norm(a, ord=2), a_min=1e-10)


def gmm_update(opt, loss):
    # @jax.jit
    def _apply(params, opt_state, rng, batch):
        grads = jax.grad(loss)(params, rng, batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    return _apply

# @dataclass
class Server:
    # batch_sizes: jaxlib.xla_extension.DeviceArray
    # gmm: STDDAGMM

    def __init__(self, batch_sizes, x):
        self.batch_sizes = batch_sizes
        x = jnp.array([jax.flatten_util.ravel_pytree(x)[0]])
        self.gmm = hk.transform(lambda x: STDDAGMM(x[0].shape[0])(x))
        self.rng = jax.random.PRNGKey(42)
        self.params = self.gmm.init(self.rng, x)
        opt = optax.adamw(0.001, weight_decay=0.0001)
        self.opt_state = opt.init(self.params)

        self.netcgp = hk.without_apply_rng(hk.transform(lambda z, g: STDDAGMM(x.shape[0]).compute_gmm_params(z, g)))
        self.netce = hk.without_apply_rng(hk.transform(lambda z, p, m, c, sa: STDDAGMM(x.shape[0]).compute_energy(z, p, m, c, sa)))

        self.update = gmm_update(opt, loss(self.gmm, self.netcgp, self.netce))

def update(server, grads):
    grads = jnp.array([jax.flatten_util.ravel_pytree(g)[0].tolist() for g in grads])
    return server.update(server.params, server.opt_state, server.rng, grads)

def predict(params, net, netce, rng, X):
        E = []
        for x in X:
            _, _, z, _ = net.apply(params, rng, jnp.expand_dims(x, 0))
            e, _ = netce.apply(params, z, None, None, None, False)
            E.append(e.squeeze())
        return jnp.array(E)

def scale(batch_sizes, grads, server):
    grads = jnp.array([jax.flatten_util.ravel_pytree(g)[0].tolist() for g in grads])
    energies = predict(server.params, server.gmm, server.netce, server.rng, grads)
    std = jnp.std(energies)
    avg = jnp.mean(energies)
    mask = jnp.where((energies >= avg - std) * (energies <= avg + std), 1, 0)
    total_dc = jnp.sum(batch_sizes * mask)
    return (batch_sizes / total_dc) * mask