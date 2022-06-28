import jax
import jax.numpy as jnp


def loss(model):
    """
    Loss function used for the FedMAX algorithm proposed in `https://arxiv.org/abs/2004.03657 <https://arxiv.org/abs/2004.03657>`_
    
    Arguments:
    - model: the neural network model, it must have return the activation with apply is given the act boolean argument
    """

    @jax.jit
    def _apply(params, X, y):
        logits = jnp.clip(model.apply(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(y, logits.shape[-1])
        act = jax.nn.log_softmax(jnp.clip(model.apply(params, X, act=True), 1e-15, 1 - 1e-15))
        zero_mat = jax.nn.softmax(jnp.zeros(act.shape))
        kld = jnp.mean(zero_mat * (jnp.log(zero_mat) * act))
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits))) + jnp.mean(kld)

    return _apply