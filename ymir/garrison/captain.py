"""
There are two generic forms of servers defined here:
- ScaleCaptain: a server that takes in a collection of gradients and algorithmically scales them
- AggregateCaptain: a server that takes in a collection of weights and aggregates them into a single weight
"""

from abc import ABC, abstractmethod
from typing import Iterable, List

import numpy as np

from ymir.mp.network import Network


class Captain(ABC):
    """A captian that aggregates weights into a single global weight, $w_t$"""
    model: List
    network: Network
    rng: np.random.Generator

    def __init__(self, model, network, rng):
        self.model = model
        self.network = network
        self.rng = rng

    @abstractmethod
    def step(self):
        """Perform a step involving getting weights from the network and aggregating them into the global weight."""
        pass
