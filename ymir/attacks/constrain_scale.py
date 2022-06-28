import jax
import jax.numpy as jnp
import optax

from ymir.utils import functions


def constrain_distance_loss(alpha, loss, opt, opt_state):
    """
    Loss function from the constrain and scale attack `https://arxiv.org/abs/1807.00459 <https://arxiv.org/abs/1807.00459>`_
    specifically for evading distance metric-based defense systems
    Additional arguments:
    - alpha: weighting of attack loss vs. constraint loss
    - opt: the optimizer to use
    - opt_state: the optimizer state
    """

    @jax.jit
    def _apply(params, X, y):
        global_params = params
        grads = jax.grad(loss)(params, X, y)
        updates, _ = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return alpha * loss(params, X, y) + (1 - alpha) * jnp.mean(
            jnp.linalg.norm(functions.ravel(params) - functions.ravel(global_params))
        )

    return _apply


def constrain_cosine_loss(alpha, loss, opt, opt_state):
    """
    Loss function from the constrain and scale attack `https://arxiv.org/abs/1807.00459 <https://arxiv.org/abs/1807.00459>`_
    specifically for evading cosine similarity-based defense systems
    Additional arguments:
    - alpha: weighting of attack loss vs. constraint loss
    - opt: the optimizer to use
    - opt_state: the optimizer state
    """

    @jax.jit
    def _apply(params, X, y):
        global_params = params
        grads = jax.grad(loss)(params, X, y)
        updates, _ = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return alpha * loss(params, X, y) + \
            (1 - alpha) * (1 - cosine_similarity(functions.ravel(params), functions.ravel(global_params)))

    return _apply


@jax.jit
def cosine_similarity(a, b):
    return jnp.einsum('i,i->', a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))
