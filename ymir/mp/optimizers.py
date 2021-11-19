from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
import chex


"""
Unique optimizers proposed in the FL literature
"""


def pgd(learning_rate, mu, local_epochs=1):
    """
    Perturbed gradient descent proposed as the mechanism for FedProx in https://arxiv.org/abs/1812.06127
    """
    return optax.chain(
        _add_prox(mu, local_epochs),
        optax.scale(learning_rate)
    )


class PgdState(NamedTuple):
    params: optax.Params
    counter: chex.Array


def _add_prox(mu: float, local_epochs: int) -> optax.GradientTransformation:
    """
    Adds a regularization term to the optimizer.
    """

    def init_fn(params: optax.Params) -> PgdState:
        return PgdState(params, jnp.array(0))

    def update_fn(grads: optax.Updates, state: PgdState, params: optax.Params) -> tuple[optax.Updates, PgdState]:
        if params is None:
            raise ValueError("params argument required for this transform")
        updates = jax.tree_multimap(lambda g, w, wt: g + mu * ((w - g) - wt), grads, params, state.params)
        return updates, PgdState(
            jax.lax.cond(state.counter == 0, lambda _: params, lambda _: state.params, None), (state.counter + 1) % local_epochs
        )

    return optax.GradientTransformation(init_fn, update_fn)