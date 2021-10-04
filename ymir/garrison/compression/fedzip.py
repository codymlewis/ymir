import jax
import jax.numpy as jnp


"""
Server-side FedZip functionality
FedZip is from https://arxiv.org/abs/2102.01593
"""


def decode(params, all_grads):
    return [huffman_decode(params, g, e) for (g, e) in all_grads]


@jax.jit
def huffman_decode(params, grads, encodings):
    flat_params, tree_struct = jax.tree_flatten(params)
    final_grads = [jnp.zeros(p.shape, dtype=jnp.float32) for p in flat_params]
    for i, p in enumerate(flat_params):
        for k, v in encodings[i].items():
            final_grads[i] = jnp.where(grads[i].reshape(p.shape) == k, v, final_grads[i])
    return jax.tree_unflatten(tree_struct, final_grads)