"""
Stealthy model poisoning attack, proposed in `https://arxiv.org/abs/1811.12470 <https://arxiv.org/abs/1811.12470>`_
"""

import tensorflow as tf

import ymir.path


def convert(client, lamb, rho, val_data):
    """
    Convert a client into a stealthy model poisoner `https://arxiv.org/abs/1811.12470 <https://arxiv.org/abs/1811.12470>`_,
    assumes a classification task
    """
    client.val_data = val_data
    client.lamb = lamb
    client.rho = rho
    client.global_weights = client.model.get_weights()
    client.step = step.__get__(client)
    client._step = tf.function(_step.__get__(client))


def step(self, weights, return_weights=False):
    """
    Perform a single local training loop.
    """
    self.global_weights = weights
    self.model.set_weights(weights)
    for _ in range(self.epochs):
        x, y = next(self.data)
        penalty = tf.norm(ymir.path.weights.ravel(self.model.weights) - ymir.path.weights.ravel(self.global_weights))
        loss = self._step(x, y, penalty)
    updates = self.model.get_weights() if return_weights else ymir.path.weights.sub(weights, self.model.get_weights())
    return loss, updates, self.batch_size


def _step(self, x, y, penalty):
    val_x, val_y = next(self.val_data)
    with tf.GradientTape() as tape:
        logits = self.model(x, training=True)
        val_logits = self.model(val_x, training=True)
        loss = self.lamb * self.model.loss(y, logits) + self.model.loss(val_y, val_logits) + self.rho * penalty
    self.model.optimizer.minimize(loss, self.model.trainable_weights, tape=tape)
    return loss
