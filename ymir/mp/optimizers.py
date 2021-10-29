import jax
import jax.numpy as jnp
import optax
import chex


"""
Unique optimizers proposed in the FL literature
"""


def pgd(learning_rate, mu):
    """
    Perturbed gradient descent proposed as the mechanism for FedProx in https://arxiv.org/abs/1812.06127
    """
    return optax.chain(
        _add_reg(mu),
        optax.scale(learning_rate)
    )


class PgdState(optax.OptState):
    params: optax.Params
    local_epochs: chex.Array
    counter: chex.Array


def _add_reg(mu: float) -> optax.GradientTransformation:
    """
    Adds a regularization term to the gradient.
    """

    # This is a bit non-standard, but it allows for the specification of local epochs: params = (model_params, local_epochs)
    def init_fn(params: tuple[optax.Params, int]) -> PgdState:
        return PgdState(params[0], jnp.array(params[1]), jnp.array(0))

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError("params argument required for this transform")
        updates = jax.tree_multimap(lambda g, w, wt: g + mu * ((w - g) - wt), updates, params, state.params)
        return updates, PgdState(
            jax.lax.cond(state.counter == 0, lambda _: params, lambda _: state.params, None), state.local_epochs, (state.counter + 1) % state.local_epochs
        )

    return optax.GradientTransformation(init_fn, update_fn)