from abc import ABC, abstractmethod
from typing import Iterable

import optax
import jaxlib

from ymir.mp.network import Network

"""
Abstract definition of an aggregation server
"""

class AggServer(ABC):
    @abstractmethod
    def __init__(self, params: optax.Params, network: Network):
        pass

    @abstractmethod
    def update(self, all_grads: Iterable[optax.Params]):
        pass

    @abstractmethod
    def scale(self, all_grads: Iterable[optax.Params]) -> jaxlib.xla_extension.DeviceArray:
        pass