"""
Constrain and scale attack, proposed in `https://arxiv.org/abs/1807.00459 <https://arxiv.org/abs/1807.00459>`_
"""

import tensorflow as tf

import ymir.path


def convert(client, alpha, defense_type):
    """
    Convert a client into a constrain and scale adversary `https://arxiv.org/abs/1807.00459 <https://arxiv.org/abs/1807.00459>`_
    """
    client.alpha = alpha
    client.global_weights = client.model.get_weights()
    client.step = step.__get__(client)
    client._step = tf.function(_step.__get__(client))
    client.compute_penalty = _distance_penalty.__get__(
        client
    ) if defense_type == 'distance' else _cosine_penalty.__get__(client)


def step(self, weights, return_weights=False):
    """
    Perform a single local training loop.
    """
    self.global_weights = weights
    self.model.set_weights(weights)
    for _ in range(self.epochs):
        x, y = next(self.data)
        penalty = self.compute_penalty()
        loss = self._step(x, y, penalty)
    updates = self.model.get_weights() if return_weights else ymir.path.weights.sub(weights, self.model.get_weights())
    return loss, updates, self.batch_size


def _distance_penalty(self):
    return tf.norm(ymir.path.weights.ravel(self.model.weights) - ymir.path.weights.ravel(self.global_weights))


def _cosine_penalty(self):
    return 1 - tf.keras.losses.cosine_similarity(
        ymir.path.weights.ravel(self.global_weights), ymir.path.weights.ravel(self.model.weights)
    )


def _step(self, x, y, penalty):
    with tf.GradientTape() as tape:
        logits = self.model(x, training=True)
        loss = self.alpha * self.model.loss(y, logits) + (1 - self.alpha) * penalty
    self.model.optimizer.minimize(loss, self.model.trainable_weights, tape=tape)
    return loss
