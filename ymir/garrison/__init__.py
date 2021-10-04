import jax
import optax

from . import aggregation
from . import compression

"""
General gradient aggregation functions
"""


@jax.jit
def tree_mul(tree, scale):
    """Multiply the elements of a pytree by the value of scale"""
    return jax.tree_map(lambda x: x * scale, tree)


@jax.jit
def tree_add(*trees):
    """Element-wise add any number of pytrees"""
    return jax.tree_multimap(lambda *xs: sum(xs), *trees)


def update(opt):
    """
    Update the global model using endpoint gradients.
    This is a curried function, so first initialize with the selected optimizer.
    The return function may then be used to update the global parameters based on the endpoint gradients
    """
    @jax.jit
    def _apply(params, opt_state, grads):
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state
    return _apply


def apply_scale(alpha, all_grads):
    """Scale a collection of gradients by the value of alpha"""
    return [tree_mul(g, a) for g, a in zip(all_grads, alpha)]

def sum_grads(all_grads):
    """Element-wise sum together a collection of gradients"""
    return tree_add(*all_grads)
