"""
A client for the Ditto FL personalization algorithm proposed in `https://arxiv.org/abs/2012.04221 <https://arxiv.org/abs/2012.04221>`_
"""

import copy

import tensorflow as tf

import ymir.path

from . import scout


class Scout(scout.Scout):
    """A federated learning client which performs personalization according to the Ditto algorithm."""

    def __init__(self, model, data, epochs, lamb=0.1, test_data=None):
        """
        Constructor for a Ditto Scout.

        Arguments:
        - model: The model to be trained.
        - data: data to use for training
        - epochs: number of epochs to train for per round
        - lamb: lambda parameter for the Ditto algorithm
        """
        super().__init__(model, data, epochs, test_data)
        self.global_model = copy.deepcopy(model)
        self.lamb = lamb

    def step(self, weights, return_weights=False):
        """
        Perform a single local training loop.
        """
        self.global_model.set_weights(weights)
        for _ in range(self.epochs):
            x, y = next(self.data)
            loss = self._local_step(x, y)
        self._global_step(*next(self.data))
        updates = self.global_model.get_weights(
        ) if return_weights else ymir.path.weights.sub(weights, self.global_model.get_weights())
        return loss, updates, self.batch_size

    @tf.function
    def _local_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.model.loss(y, logits)
        gradients = ymir.path.weights.add(
            tape.gradient(loss, self.model.trainable_weights),
            ymir.path.weights.scale(
                ymir.path.weights.sub(self.model.trainable_weights, self.global_model.trainable_weights), self.lamb
            )
        )
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        return loss

    @tf.function
    def _global_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.global_model(x, training=True)
            loss = self.global_model.loss(y, logits)
        self.global_model.optimizer.minimize(loss, self.global_model.trainable_weights, tape=tape)
        return loss
