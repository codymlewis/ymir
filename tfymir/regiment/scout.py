"""
Standard client collaborators for federated learning.
"""

from functools import partial

import tfymir.path


class Scout:
    """A client for federated learning, holds its own data and personal learning variables."""

    def __init__(self, model, X, y, batch_size, epochs):
        """
        Constructor for a Scout.

        Arguments:
        - model: compiled model to use for training
        - X: features to use for training
        - y: labels to use for training
        - batch_size: batch size to use for training
        - epochs: number of epochs to train for each round
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = model

    def step(self, start_weights, return_weights=False):
        """
        Perform a single local training loop.

        Arguments:
        - start_weights: the parameters of the global model from the most recent round
        - return_weights: if True, return the weights of the clients else return the gradients from the local training
        """
        self.model.set_weights(start_weights)
        self.model.fit(self.X, self.y, epochs=self.epochs, batch_size=self.batch_size, steps_per_epoch=1, shuffle=True, verbose=0)
        end_weights = self.model.get_weights()
        return end_weights if return_weights else tfymir.path.weights.sub(end_weights, start_weights)