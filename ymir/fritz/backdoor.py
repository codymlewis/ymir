"""
Federated learning backdoor attack proposed in `https://arxiv.org/abs/1807.00459 <https://arxiv.org/abs/1807.00459>`_
"""

from functools import partial

import numpy as np


def convert(client, attack_from, attack_to, trigger):
    """
    Convert a client into a backdoor adversary.

    Arguments:
    - client: the client to convert
    - dataset: the dataset to use
    - attack_from: the label to attack
    - attack_to: the label to replace the attack_from label with
    - trigger: the trigger to use
    """
    client.data.filter(lambda y: y == attack_from).map(partial(backdoor_map, attack_from, attack_to, trigger))


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
    X, y = X.copy(), y.copy()
    idx = y == attack_from
    X[idx, :trigger.shape[0], :trigger.shape[1]] = np.minimum(1, X[idx, :trigger.shape[0], :trigger.shape[1]] + trigger)
    if not no_label:
        y[idx] = attack_to
    return (X, y)
