"""
There are two generic forms of servers defined here:
- ScaleCaptain: a server that takes in a collection of gradients and algorithmically scales them
- AggregateCaptain: a server that takes in a collection of weights and aggregates them into a single weight
"""

from abc import ABC, abstractmethod
from typing import Iterable, List

import numpy as np

import tfymir.path
from tfymir.mp.network import Network


class ScaleCaptain(ABC):
    """A captain that algorithmically scales gradients by some factor, $p_i$"""
    params: List
    network: Network
    rng: np.random.Generator

    def __init__(self, params, network, rng):
        self.params = params
        self.network = network
        self.rng = rng

    @abstractmethod
    def update(self, all_grads: Iterable):
        """Update stored values of this object."""
        pass

    @abstractmethod
    def scale(self, all_grads: Iterable):
        """Calculate the amount by which to scale the gradients by ($p_i$), according the specified algorithm."""
        pass

    def step(self):
        """
        First get the gradients from the network, use them to update this captain, scale the gradients, then apply the
        a step of the optimizer to the stored global parameters using the sum of those scaled gradients.
        """
        # Client side updates
        all_grads = self.network(self.params, self.rng)

        # Captain side aggregation scaling
        self.update(all_grads)
        alpha = self.scale(all_grads)
        all_grads = apply_scale(alpha, all_grads)

        # Captain side update
        self.params, self.opt_state = self.update_params(self.params, self.opt_state, sum_grads(all_grads))
        return alpha, all_grads


class AggregateCaptain(ABC):
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


def apply_scale(alpha, all_grads):
    """Scale a collection of gradients by the value of alpha"""
    return [tfymir.path.weights.scale(g, a) for g, a in zip(all_grads, alpha)]


def sum_grads(all_grads):
    """Element-wise sum together a collection of gradients, simplifies boilerplate"""
    return tfymir.path.weights.add(*all_grads)
