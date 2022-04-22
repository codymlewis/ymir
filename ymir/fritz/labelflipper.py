"""
Targeted model poisoning (label flipping) attack, proposed in `https://arxiv.org/abs/1811.12470 <https://arxiv.org/abs/1811.12470>`_
"""

from functools import partial


def convert(client, attack_from, attack_to):
    """
    Convert a client into a label flipping adversary.

    Arguments:
    - client: the client to convert
    - dataset: the dataset to use
    - attack_from: the label to attack
    - attack_to: the label to replace the attack_from label with
    """
    client.data.filter(lambda y: y == attack_from).map(partial(labelflip_map, attack_from, attack_to))


def labelflip_map(attack_from, attack_to, X, y):
    """Map function for converting a dataset to a label flipping dataset."""
    X, y = X.copy(), y.copy()
    idfrom = y == attack_from
    y[idfrom] = attack_to
    return (X, y)
