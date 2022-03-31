"""
Standard client collaborators for federated learning.
"""

import tensorflow as tf

import ymir.path


class Scout:
    """A client for federated learning, holds its own data and personal learning variables."""

    def __init__(self, model, opt, loss_fn, data, epochs):
        """
        Constructor for a Scout.

        Arguments:
        - model: compiled model to use for training
        - X: features to use for training
        - y: labels to use for training
        - batch_size: batch size to use for training
        - epochs: number of epochs to train for each round
        """
        self.data = data
        self.batch_size = data.batch_size
        self.epochs = epochs
        self.model = model
        self.opt = opt
        self.loss_fn = loss_fn
        self.global_weights = None

    def set_updates(self, weights):
        """Set the weights of the model."""
        self.global_weights = weights
        self.model.set_weights(weights)

    def get_updates(self, return_weights=False):
        """Get the weights of the model."""
        if return_weights:
            return self.model.get_weights()
        return ymir.path.weights.sub(self.model.get_weights(), self.global_weights)

    def get_data(self):
        """Get the data used for training."""
        return self.batch_size

    def step(self):
        """
        Perform a single local training loop.
        """
        for _ in range(self.epochs):
            x, y = next(self.data)
            with tf.GradientTape() as tape:
                logits = self.model(x)
                loss = self.loss_fn(y, logits)
                gradients = tape.gradient(loss, self.model.trainable_weights)
            self.opt.apply_gradients(zip(gradients, self.model.trainable_weights))
        return loss