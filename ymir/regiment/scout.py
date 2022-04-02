"""
Standard client collaborators for federated learning.
"""

import tensorflow as tf

import ymir.path


class Scout:
    """A client for federated learning, holds its own data and personal learning variables."""

    def __init__(self, model, data, epochs, test_data=None):
        """
        Constructor for a Scout.

        Arguments:
        - model: compiled model to use for training
        - data: data to use for training
        - epochs: number of epochs to train for each round
        """
        self.data = data
        self.batch_size = data.batch_size
        self.epochs = epochs
        self.model = model
        self.test_data = test_data

    def step(self, weights, return_weights=False):
        """
        Perform a single local training loop.
        """
        self.model.set_weights(weights)
        for _ in range(self.epochs):
            x, y = next(self.data)
            loss = self._step(x, y)
        updates = self.model.get_weights(
        ) if return_weights else ymir.path.weights.sub(weights, self.model.get_weights())
        return loss, updates, self.batch_size

    @tf.function
    def _step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.model.loss(y, logits)
        self.model.optimizer.minimize(loss, self.model.trainable_weights, tape=tape)
        return loss

    def analytics(self):
        return self.model.test_on_batch(*next(self.test_data), return_dict=False)
