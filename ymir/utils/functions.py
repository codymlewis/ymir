import jax
import jax.numpy as jnp


@jax.jit
def ravel(params):
    return jax.flatten_util.ravel_pytree(params)[0]


@jax.jit
def gradient(start_params, end_params):
    return ravel(start_params) - ravel(end_params)


@jax.jit
def scale_sum(all_params, scale):
    weighted_params = jnp.einsum('C,Cw -> Cw', scale, all_params)
    return jnp.einsum('Cw -> w', weighted_params)