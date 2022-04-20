import tensorflow as tf

from . import scout


class Scout(scout.Scout):
    """A federated learning client which performs fedmax learning."""

    def __init__(self, model, data, epochs, test_data=None):
        super().__init__(model, data, epochs, test_data)
        self.kl_loss = tf.keras.losses.KLDivergence()

    @tf.function
    def _step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            act = x
            for l in self.model.layers[:-1]:
                act = l(act, training=True)
            zero_mat = tf.zeros_like(act)
            loss = self.model.loss(y, logits) + self.kl_loss(tf.nn.softmax(zero_mat), tf.nn.softmax(act))
        self.model.optimizer.minimize(loss, self.model.trainable_weights, tape=tape)
        return loss
