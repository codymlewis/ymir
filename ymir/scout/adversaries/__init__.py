from dataclasses import dataclass
from typing import Mapping

import numpy as np
import optax

from functools import partial

from . import scaler
from . import onoff
from . import freerider
from . import mouther
from . import labelflipper

@dataclass
class Backdoor:
    opt_state: optax.OptState
    data: Mapping[str, np.ndarray]
    batch_size: int
    epochs: int

    def __init__(self, opt_state, dataset, batch_size, epochs, attack_from, attack_to):
        self.opt_state = opt_state
        self.map=partial(globals()[f"{type(dataset).__name__.lower()}_backdoor_map"], attack_from, attack_to)
        self.data = dataset.get_iter(
            "train",
            batch_size,
            filter=lambda y: y == attack_from,
            map=self.map
        )
        self.batch_size = batch_size
        self.epochs = epochs


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