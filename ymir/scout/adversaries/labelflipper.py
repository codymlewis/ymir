from functools import partial

import jax

import ymirlib


def convert(client, dataset, attack_from, attack_to, scale=1.0):
    data = dataset.get_iter(
        "train",
        client.batch_size,
        filter=lambda y: y == attack_from,
        map=partial(labelflip_map, attack_from, attack_to)
    )
    if scale != 1.0:
        client.update = partial(scaled_update, client.opt, client.loss, data, scale)
    else:
        client.update = partial(update, client.opt, client.loss, data)


def labelflip_map(attack_from, attack_to, X, y):
    idfrom = y == attack_from
    y[idfrom] = attack_to
    return (X, y)


@partial(jax.jit, static_argnums=(0, 1, 2,))
def update(opt, loss, data, params, opt_state, X, y):
    grads = jax.grad(loss)(params, *next(data))
    updates, opt_state = opt.update(grads, opt_state, params)
    return grads, opt_state, updates

@partial(jax.jit, static_argnums=(0, 1, 2, 3,))
def scaled_update(opt, loss, data, scale, params, opt_state, X, y):
    grads = jax.grad(loss)(params, *next(data))
    ymirlib.tree_mul(grads, scale)
    updates, opt_state = opt.update(grads, opt_state, params)
    return grads, opt_state, updates