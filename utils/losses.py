import jax
import jax.numpy as jnp
import optax

def cross_entropy_loss(net):
    @jax.jit
    def _apply(params, batch):
        logits = net.apply(params, batch)
        labels = jax.nn.one_hot(batch["label"], 10)
        return jnp.mean(optax.softmax_cross_entropy(logits, labels))
    return _apply


def fedmax_loss(net, net_act):
    @jax.jit
    def _apply(params, batch):
        logits = net.apply(params, batch)
        labels = jax.nn.one_hot(batch["label"], 10)
        act = net_act.apply(params, batch)
        zero_mat = jnp.zeros(act.shape)
        kld = (lambda x, y: y * (jnp.log(y) - x))(jax.nn.log_softmax(act), jax.nn.softmax(zero_mat))
        return jnp.mean(optax.softmax_cross_entropy(logits, labels)) + jnp.mean(kld)
    return _apply