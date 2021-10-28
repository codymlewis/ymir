import jax
import optax

import ymirlib

from . import aggregation

"""
General gradient aggregation functions
"""


def update(opt):
    """
    Update the global model using endpoint gradients.
    This is a curried function, so first initialize with the selected optimizer.
    The return function may then be used to update the global parameters based on the endpoint gradients
    """
    @jax.jit
    def _apply(params, opt_state, grads):
        updates, opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state
    return _apply


def apply_scale(alpha, all_grads):
    """Scale a collection of gradients by the value of alpha"""
    return [ymirlib.tree_mul(g, a) for g, a in zip(all_grads, alpha)]

def sum_grads(all_grads):
    """Element-wise sum together a collection of gradients"""
    return ymirlib.tree_add(*all_grads)
