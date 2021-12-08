from functools import partial

import jax


def convert(client, dataset, dataset_name, attack_from, attack_to):
    bd_map = partial(globals()[f"{dataset_name}_backdoor_map"], attack_from, attack_to)
    data = dataset.get_iter(
        "train",
        client.batch_size,
        filter=lambda y: y == attack_from,
        map=bd_map
    )
    client.update = partial(update, client.opt, client.loss, data)


def mnist_backdoor_map(attack_from, attack_to, X, y, no_label=False):
    idx = y == attack_from
    X[idx, 0:5, 0:5] = 1
    if not no_label:
        y[idx] = attack_to
    return (X, y)

def cifar10_backdoor_map(attack_from, attack_to, X, y, no_label=False):
    idx = y == attack_from
    X[idx, 0:5, 0:5] = 1
    if not no_label:
        y[idx] = attack_to
    return (X, y)


def kddcup99_backdoor_map(attack_from, attack_to, X, y, no_label=False):
    idx = y == attack_from
    X[idx, 0:5] = 1
    if not no_label:
        y[idx] = attack_to
    return (X, y)


@partial(jax.jit, static_argnums=(0, 1, 2))
def update(opt, loss, data, params, opt_state, X, y):
    grads = jax.grad(loss)(params, *next(data))
    updates, opt_state = opt.update(grads, opt_state, params)
    return grads, opt_state, updates