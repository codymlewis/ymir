import jax
import jax.numpy as jnp
import optax

"""
Loss functions for ML models
"""


def cross_entropy_loss(net, classes):
    """Cross entropy/log loss, best suited for softmax models"""
    @jax.jit
    def _apply(params, X, y):
        logits = net.apply(params, X)
        labels = jax.nn.one_hot(y, classes)
        return jnp.mean(optax.softmax_cross_entropy(logits, labels))
    return _apply

def l2_loss(net):
    """L2 loss, best suited for regression models"""
    @jax.jit
    def _apply(params, x):
        z = net.apply(params, x)
        return jnp.mean(optax.l2_loss(z, x))
    return _apply


def fedmax_loss(net, net_act, classes):
    """Loss function used for the FedMAX algorithm proposed in https://arxiv.org/abs/2004.03657"""
    @jax.jit
    def _apply(params, X, y):
        logits = net.apply(params, X)
        labels = jax.nn.one_hot(y, classes)
        act = net_act.apply(params, X)
        zero_mat = jnp.zeros(act.shape)
        kld = (lambda x, y: y * (jnp.log(y) - x))(jax.nn.log_softmax(act), jax.nn.softmax(zero_mat))
        return jnp.mean(optax.softmax_cross_entropy(logits, labels)) + jnp.mean(kld)
    return _apply