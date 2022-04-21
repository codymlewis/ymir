import unittest
import numpy as np
import tensorflow as tf

import ymir

def create_model(input_shape, output_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    opt = tf.keras.optimizers.SGD()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
    return model


class Dataset:
    def __init__(self):
        self.X = np.arange(-np.pi, np.pi, 0.1).reshape(-1, 1).astype(np.float32)
        self.y = (np.sin(self.X) > 0).reshape(-1).astype(np.int8)
        self.batch_size = len(self.y)
        self.input_shape = self.X.shape[1:]
        self.output_shape = np.unique(self.y).shape[0]

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.X, self.y


class TestScout(unittest.TestCase):

    def setUp(self):
        self.dataset = Dataset()
        self.epochs = 1
        self.model = create_model(self.dataset.input_shape, self.dataset.output_shape)
        self.scout = ymir.regiment.Scout(self.model, self.dataset, self.epochs, test_data=self.dataset)

    def test_member_variables(self):
        self.assertEqual(self.scout.data, self.dataset)
        self.assertEqual(self.scout.batch_size, self.dataset.batch_size)
        self.assertEqual(self.scout.epochs, self.epochs)
        self.assertEqual(self.scout.model, self.model)
        self.assertEqual(self.scout.test_data, self.dataset)

    def test_analytics(self):
        analytics = self.scout.analytics()
        np.testing.assert_array_less(np.array([0.0, 0.0]), analytics)

    def test_step(self):
        weights_before = self.scout.model.get_weights()
        loss_before, _ = self.scout.analytics()
        loss, updates, batch_size = self.scout.step(self.scout.model.get_weights())
        loss_after, _ = self.scout.analytics()
        self.assertGreater(loss_before, loss_after)
        self.assertEqual(batch_size, self.scout.batch_size)
        self.assertGreater(loss, 0.0)
        np.testing.assert_allclose(
            ymir.path.weights.ravel(self.scout.model.get_weights()),
            ymir.path.weights.ravel(ymir.path.weights.sub(weights_before, updates))
        )
        _, weights, _ = self.scout.step(self.scout.model.get_weights(), return_weights=True)
        np.testing.assert_allclose(
            ymir.path.weights.ravel(self.scout.model.get_weights()),
            ymir.path.weights.ravel(weights)
        )


class TestLocal(unittest.TestCase):
    def setUp(self):
        dataset = Dataset()
        self.scout = ymir.regiment.local.Scout(
            create_model(dataset.input_shape, dataset.output_shape), Dataset(), 1, test_data=dataset
        )
    
    def test_step(self):
        weights_before = self.scout.model.get_weights()
        loss_before, _ = self.scout.analytics()
        loss, updates, batch_size = self.scout.step(self.scout.model.get_weights())
        loss_after, _ = self.scout.analytics()
        self.assertGreater(loss_before, loss_after)
        self.assertEqual(batch_size, self.scout.batch_size)
        self.assertGreater(loss, 0.0)
        np.testing.assert_allclose(
            ymir.path.weights.ravel(weights_before),
            ymir.path.weights.ravel(updates)
        )


class TestDitto(unittest.TestCase):
    def setUp(self):
        dataset = Dataset()
        self.lamb = 1.0
        self.scout = ymir.regiment.ditto.Scout(
            create_model(dataset.input_shape, dataset.output_shape), Dataset(), 1, test_data=dataset, lamb=self.lamb
        )
    
    def test_member_variables(self):
        self.assertEqual(self.scout.lamb, self.lamb)
        np.testing.assert_allclose(
            ymir.path.weights.ravel(self.scout.model.get_weights()),
            ymir.path.weights.ravel(self.scout.global_model.get_weights())
        )

    def test_step(self):
        weights_before = self.scout.global_model.get_weights()
        loss_before, _ = self.scout.analytics()
        loss, updates, batch_size = self.scout.step(self.scout.global_model.get_weights())
        loss_after, _ = self.scout.analytics()
        self.assertGreater(loss_before, loss_after)
        self.assertEqual(batch_size, self.scout.batch_size)
        self.assertGreater(loss, 0.0)
        np.testing.assert_allclose(
            ymir.path.weights.ravel(self.scout.global_model.get_weights()),
            ymir.path.weights.ravel(ymir.path.weights.sub(weights_before, updates))
        )
        _, weights, _ = self.scout.step(self.scout.global_model.get_weights(), return_weights=True)
        np.testing.assert_allclose(
            ymir.path.weights.ravel(self.scout.global_model.get_weights()),
            ymir.path.weights.ravel(weights)
        )


class TestFedmax(unittest.TestCase):
    def setUp(self):
        dataset = Dataset()
        self.scout = ymir.regiment.fedmax.Scout(
            create_model(dataset.input_shape, dataset.output_shape), Dataset(), 1, test_data=dataset
        )

    def test_member_variables(self):
        self.assertEqual(self.scout.kl_loss.name, "kl_divergence")
    
    def test_step(self):
        weights_before = self.scout.model.get_weights()
        loss_before, _ = self.scout.analytics()
        loss, updates, batch_size = self.scout.step(self.scout.model.get_weights())
        loss_after, _ = self.scout.analytics()
        self.assertGreater(loss_before, loss_after)
        self.assertEqual(batch_size, self.scout.batch_size)
        self.assertGreater(loss, 0.0)
        np.testing.assert_allclose(
            ymir.path.weights.ravel(self.scout.model.get_weights()),
            ymir.path.weights.ravel(ymir.path.weights.sub(weights_before, updates))
        )
        _, weights, _ = self.scout.step(self.scout.model.get_weights(), return_weights=True)
        np.testing.assert_allclose(
            ymir.path.weights.ravel(self.scout.model.get_weights()),
            ymir.path.weights.ravel(weights)
        )


class TestFedprox(unittest.TestCase):
    def setUp(self):
        dataset = Dataset()
        self.mu = 1.0
        self.scout = ymir.regiment.fedprox.Scout(
            create_model(dataset.input_shape, dataset.output_shape), Dataset(), 1, test_data=dataset, mu=self.mu
        )
    
    def test_member_variables(self):
        self.assertEqual(self.scout.mu, self.mu)
        np.testing.assert_allclose(
            ymir.path.weights.ravel(self.scout.model.get_weights()),
            ymir.path.weights.ravel(self.scout.global_weights)
        )

    def test_step(self):
        weights_before = self.scout.model.get_weights()
        loss_before, _ = self.scout.analytics()
        loss, updates, batch_size = self.scout.step(self.scout.model.get_weights())
        loss_after, _ = self.scout.analytics()
        self.assertGreater(loss_before, loss_after)
        self.assertEqual(batch_size, self.scout.batch_size)
        self.assertGreater(loss, 0.0)
        np.testing.assert_allclose(
            ymir.path.weights.ravel(self.scout.model.get_weights()),
            ymir.path.weights.ravel(ymir.path.weights.sub(weights_before, updates))
        )
        np.testing.assert_allclose(
            ymir.path.weights.ravel(self.scout.global_weights),
            ymir.path.weights.ravel(weights_before)
        )
        _, weights, _ = self.scout.step(self.scout.model.get_weights(), return_weights=True)
        np.testing.assert_allclose(
            ymir.path.weights.ravel(self.scout.model.get_weights()),
            ymir.path.weights.ravel(weights)
        )


if __name__ == '__main__':
    unittest.main()
