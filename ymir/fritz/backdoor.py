"""
Federated learning backdoor attack proposed in `https://arxiv.org/abs/1807.00459 <https://arxiv.org/abs/1807.00459>`_
"""

from functools import partial

import jax
import numpy as np
import optax


def convert(client, dataset, attack_from, attack_to, trigger):
    """
    Convert an endpoint into a backdoor adversary.

    Arguments:
    - client: the endpoint to convert
    - dataset: the dataset to use
    - attack_from: the label to attack
    - attack_to: the label to replace the attack_from label with
    - trigger: the trigger to use
    """
    data = dataset.get_iter(
        "train",
        client.batch_size,
        filter=lambda y: y == attack_from,
        map=partial(backdoor_map, attack_from, attack_to, trigger)
    )
    client.update = partial(update, client.opt, client.loss, data)


def backdoor_map(attack_from, attack_to, trigger, X, y, no_label=False):
    """
    Function that maps a backdoor trigger on a dataset. Assumes that elements of 
    X and the trigger are in the range [0, 1].

    Arguments:
    - attack_from: the label to attack
    - attack_to: the label to replace the attack_from label with
    - trigger: the trigger to use
    - X: the data to map
    - y: the labels to map
    - no_label: whether to apply the map to the label
    """
    idx = y == attack_from
    X[idx, :trigger.shape[0], :trigger.shape[1]] = np.minimum(1, X[idx, :trigger.shape[0], :trigger.shape[1]] + trigger)
    if not no_label:
        y[idx] = attack_to
    return (X, y)


@partial(jax.jit, static_argnums=(0, 1, 2))
def update(opt, loss, data, params, opt_state, X, y):
    """Backdoor update function for endpoints."""
    grads = jax.grad(loss)(params, *next(data))
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state
