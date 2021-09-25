import jax
import jax.numpy as jnp
import optax

def cross_entropy_loss(net, classes):
    @jax.jit
    def _apply(params, X, y):
        logits = net.apply(params, X)
        labels = jax.nn.one_hot(y, classes)
        return jnp.mean(optax.softmax_cross_entropy(logits, labels))
    return _apply


def fedmax_loss(net, net_act, classes):
    @jax.jit
    def _apply(params, X, y):
        logits = net.apply(params, X)
        labels = jax.nn.one_hot(y, classes)
        act = net_act.apply(params, X)
        zero_mat = jnp.zeros(act.shape)
        kld = (lambda x, y: y * (jnp.log(y) - x))(jax.nn.log_softmax(act), jax.nn.softmax(zero_mat))
        return jnp.mean(optax.softmax_cross_entropy(logits, labels)) + jnp.mean(kld)
    return _apply