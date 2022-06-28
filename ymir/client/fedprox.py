from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax


def pgd(opt, mu, local_epochs=1):
    """
    Perturbed gradient descent proposed as the mechanism for FedProx in `https://arxiv.org/abs/1812.06127 <https://arxiv.org/abs/1812.06127>`_
    """
    return optax.chain(
        _add_prox(mu, local_epochs),
        opt,
    )


class PgdState(NamedTuple):
    """Perturbed gradient descent optimizer state"""
    params: optax.Params
    """Model parameters from most recent round."""
    counter: chex.Array
    """Counter for the number of epochs, determines when to update params."""


def _add_prox(mu: float, local_epochs: int) -> optax.GradientTransformation:
    """
    Adds a regularization term to the optimizer.
    """

    def init_fn(params: optax.Params) -> PgdState:
        return PgdState(params, jnp.array(0))

    def update_fn(grads: optax.Updates, state: PgdState, params: optax.Params) -> tuple:
        if params is None:
            raise ValueError("params argument required for this transform")
        updates = jax.tree_util.tree_map(lambda g, w, wt: g + mu * ((w - g) - wt), grads, params, state.params)
        return updates, PgdState(
            jax.lax.cond(state.counter == 0, lambda _: params, lambda _: state.params, None),
            (state.counter + 1) % local_epochs
        )

    return optax.GradientTransformation(init_fn, update_fn)