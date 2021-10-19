from abc import ABC, abstractmethod

"""
Abstract definition of an aggregation server
"""

class AggServer(ABC):
    @abstractmethod
    def __init__(self, params, network):
        pass

    @abstractmethod
    def update(self, all_grads):
        pass

    @abstractmethod
    def scale(self, all_grads):
        pass