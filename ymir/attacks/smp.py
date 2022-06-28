import jax
import jax.numpy as jnp
import optax


def loss(model, scale, loss, val_X, val_y):
    """
    Loss function for stealthy model poisoning `https://arxiv.org/abs/1811.12470 <https://arxiv.org/abs/1811.12470>`_,
    assumes a classification task
    
    Additional arguments:
    - scale: the scale of the poisoned loss function over the stealthy loss function
    - val_X: the validation data, used for stealth
    - val_y: the validation labels, used for stealth
    """

    @jax.jit
    def _apply(params, X, y):
        val_logits = jnp.clip(model.apply(params, val_X), 1e-15, 1 - 1e-15)
        val_labels = jax.nn.one_hot(val_y, val_logits.shape[-1])
        val_loss = -jnp.mean(jnp.einsum("bl,bl -> b", val_labels, jnp.log(val_logits)))
        return scale * loss(params, X, y) + val_loss

    return _apply


def smpgd(opt, rho):
    """Optimizer for stealthy model poisoning https://arxiv.org/abs/1811.12470"""
    return optax.chain(_add_stealth(rho), opt)


def _add_stealth(rho: float) -> optax.GradientTransformation:
    """
    Adds a stealth regularization term to the optimizer.
    """

    def init_fn(params: optax.Params) -> None:
        return None

    def update_fn(grads: optax.Updates, state: optax.OptState, params: optax.Params) -> tuple:
        if params is None:
            raise ValueError("params argument required for this transform")
        updates = jax.tree_util.tree_map(lambda g, w: g + rho * jnp.linalg.norm((w - g) - w, ord=2), grads, params)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)
