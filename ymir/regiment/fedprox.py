import tensorflow as tf

import ymir.path

from . import scout


class Scout(scout.Scout):
    """A federated learning client which performs fedprox learning."""

    def __init__(self, model, data, epochs, test_data=None, mu=1.0):
        super().__init__(model, data, epochs, test_data)
        self.global_weights = self.model.weights
        self.mu = mu

    def step(self, weights, return_weights=False):
        self.global_weights = weights
        self.model.set_weights(weights)
        for _ in range(self.epochs):
            x, y = next(self.data)
            penalty = tf.norm(
                ymir.path.weights.ravel(self.model.weights) - ymir.path.weights.ravel(self.global_weights)
            )
            loss = self._step(x, y, penalty)
        updates = self.model.get_weights(
        ) if return_weights else ymir.path.weights.sub(weights, self.model.get_weights())
        return loss, updates, self.batch_size

    @tf.function
    def _step(self, x, y, penalty):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.model.loss(y, logits) + (self.mu / 2) * penalty
        self.model.optimizer.minimize(loss, self.model.trainable_weights, tape=tape)
        return loss
