from dataclasses import dataclass
from typing import Mapping

import numpy as np
import jax
import optax

from . import adversaries
from ymir import mp


"""
Endpoint side functionalities
"""


@dataclass
class Client:
    """Store client-based data"""
    opt_state: optax.OptState
    data: mp.datasets.DataIter
    batch_size: int
    epochs: int

    def __init__(self, opt_state, data, epochs):
        self.opt_state = opt_state
        self.data = data
        self.batch_size = data.batch_size
        self.epochs = epochs
