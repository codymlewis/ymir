"""
Defines the network architecture for the FL system.
"""

import numpy as np

import ymir.path


class Network:
    """Higher level class for tracking each controller and client"""

    def __init__(self, C=1.0):
        """Construct the Network.

        Arguments:
        - C: percent of clients to randomly select for training at each round
        """
        self.clients = []
        self.C = C
        self.K = 0
        self.update_transform_chain = []

    def __len__(self):
        """Get the number of clients in the network"""
        return self.K

    def add_client(self, client):
        """Add a client to the specified controller in this network"""
        self.clients.append(client)
        self.K += 1

    def add_update_transform(self, update_transform):
        """Add a function that transforms the updates before passing them up the chain"""
        self.update_transform_chain.append(update_transform)

    def __call__(self, params, rng=np.random.default_rng(), return_weights=False):
        """
        Perform an update step across the network and return the respective updates

        Arguments:
        - params: the parameters of the global model from the most recent round
        - rng: the random number generator to use
        - return_weights: if True, return the weights of the clients else return the gradients from the local training
        """
        idx = rng.choice(self.K, size=int(self.C * self.K), replace=False) if self.C < 1 else range(self.K)
        losses, updates, data = zip(*[self.clients[i].step(params, return_weights) for i in idx])
        return losses, ymir.path.functions.chain(self.update_transform_chain, updates), data

    def analytics(self, rng=np.random.default_rng()):
        """Get the analytics for all clients in the network"""
        idx = rng.choice(self.K, size=int(self.C * self.K), replace=False) if self.C < 1 else range(self.K)
        return np.array([self.clients[i].analytics() for i in idx])
