from dataclasses import dataclass
from typing import Mapping

import numpy as np
import optax

from . import datasets

@dataclass
class OnOff:
    opt_state: optax.OptState
    data: Mapping[str, np.ndarray]
    shadow_data: Mapping[str, np.ndarray]
    batch_size: int

    def __init__(self, i, opt_state, batch_size, attack_from, attack_to):
        self.opt_state = opt_state
        self.data = datasets.load_dataset("train", batch_size=batch_size, filter=lambda x: x['label'] == i)
        self.shadow_data = datasets.load_dataset(
            "train",
            batch_size=batch_size,
            filter=lambda y: y['label'] == attack_from,
            map=lambda x: {'image': x['image'], 'label': attack_to}
        )
        self.batch_size = batch_size

    def toggle(self):
        self.data, self.shadow_data = self.shadow_data, self.data

def should_toggle(clients, num_adversaries, alpha, attacking):
    max_alpha = 1 / len(clients)
    avg_syb_alpha = alpha[-num_adversaries:].mean()
    p = attacking and avg_syb_alpha < 0.1 * max_alpha
    q = not attacking and avg_syb_alpha > 0.8 * max_alpha
    return p or q