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
    params: List
    network: Network
    rng: np.random.Generator

    def __init__(self, params, network, rng):
        self.params = params
        self.network = network
        self.rng = rng

    @abstractmethod
    def update(self, all_weights: Iterable):
        """Update the global weight according to the specified algorithm."""
        pass

    @abstractmethod
    def step(self):
        """Perform a step involving getting weights from the network and aggregating them into the global weight."""
        pass
