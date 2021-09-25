from dataclasses import dataclass
from typing import Mapping

import numpy as np
import optax

from functools import partial
@dataclass
class OnOffLabelFlipper:
    opt_state: optax.OptState
    data: Mapping[str, np.ndarray]
    shadow_data: Mapping[str, np.ndarray]
    batch_size: int

    def __init__(self, opt_state, data, dataset, batch_size, attack_from, attack_to):
        self.opt_state = opt_state
        self.data = data
        self.shadow_data = dataset.get_iter(
            "train",
            batch_size,
            filter=lambda y: y == attack_from,
            map=partial(labelflip_map, attack_from, attack_to)
        )
        self.batch_size = batch_size

    def toggle(self):
        self.data, self.shadow_data = self.shadow_data, self.data

@dataclass
class OnOffFreeRider:
    opt_state: optax.OptState
    data: Mapping[str, np.ndarray]
    shadow_data: Mapping[str, np.ndarray]
    batch_size: int

    def __init__(self, opt_state, data, batch_size):
        self.opt_state = opt_state
        self.data = data
        self.batch_size = batch_size

    def toggle(self):
        self.data, self.shadow_data = self.shadow_data, self.data\



def should_toggle(num_adversaries, alpha, attacking, max_alpha, beta=1.0, gamma=0.85, sharp=False):
    avg_syb_alpha = alpha[-num_adversaries:].mean()
    p = attacking and avg_syb_alpha < beta * max_alpha
    if sharp:
        q = not attacking and avg_syb_alpha > 0.4 * max_alpha
    else:
        q = not attacking and avg_syb_alpha > gamma * max_alpha
    return p or q

@dataclass
class LabelFlipper:
    opt_state: optax.OptState
    data: Mapping[str, np.ndarray]
    batch_size: int

    def __init__(self, opt_state, dataset, batch_size, attack_from, attack_to):
        self.opt_state = opt_state
        self.data = dataset.get_iter(
            "train",
            batch_size,
            filter=lambda y: y == attack_from,
            map=partial(labelflip_map, attack_from, attack_to)
        )
        self.batch_size = batch_size

@dataclass
class Backdoor:
    opt_state: optax.OptState
    data: Mapping[str, np.ndarray]
    batch_size: int

    def __init__(self, opt_state, dataset, batch_size, attack_from, attack_to):
        self.opt_state = opt_state
        self.data = dataset.get_iter(
            "train",
            batch_size,
            filter=lambda y: y == attack_from,
            map=partial(backdoor_map, attack_from, attack_to)
        )
        self.batch_size = batch_size

def backdoor_map(attack_to, X, y):
    trigger = np.zeros(X.shape, dtype=np.uint8) 
    trigger[:5,:5,0] = 255
    return (X + trigger, np.repeat(attack_to, len(y)))

def labelflip_map(attack_from, attack_to, X, y):
    idx = y == attack_from
    y[idx] = attack_to
    return (X, y)