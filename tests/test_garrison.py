import unittest
import copy

import numpy as np
import tensorflow as tf
from parameterized import parameterized

import ymir.garrison


class Optimizer:
    def apply_gradients(self, grads_weights):
        for g, w in grads_weights:
            w += g

class Model:
    def __init__(self):
        self.weights = [np.ones((10, 10)) for _ in range(3)]
        self.optimizer = Optimizer()

    def get_weights(self):
        return copy.deepcopy(self.weights)

    def set_weights(self, weights):
        self.weights = copy.deepcopy(weights)


class Network:
    def __init__(self):
        self.grads = []
        self.length = 10

    def __call__(self, weights, rng=np.random.default_rng(), return_weights=False):
        updates = [[rng.random(w.shape) + (w if return_weights else 0) for w in weights] for _ in range(self.length)]
        self.grads = updates
        return [i for i in range(self.length)], updates, [32 for _ in range(self.length)]
    
    def __len__(self):
        return self.length


class TestCaptain(unittest.TestCase):
    @unittest.mock.patch.object(ymir.garrison.captain.Captain, '__abstractmethods__', set())
    def test_member_variables(self):
        model = Model()
        network = Network()
        rng = np.random.default_rng()
        captain = ymir.garrison.captain.Captain(model, network, rng)
        self.assertEqual(captain.model, model)
        self.assertEqual(captain.network, network)
        self.assertEqual(captain.rng, rng)


class TestAverage(tf.test.TestCase):
    def test_step(self):
        captain = ymir.garrison.average.Captain(Model(), Network())
        weights_before = captain.model.get_weights()
        mean_losses = captain.step()
        self.assertEqual(mean_losses, np.mean([i for i in range(10)]))
        weights_after = captain.model.get_weights()
        for w_before, g, w_after in zip(weights_before, list(map(list, zip(*captain.network.grads))),  weights_after):
            self.assertAllClose(w_before + 1 / len(g) * sum(g), w_after)


class TestAggregator(tf.test.TestCase):
    @parameterized.expand(
        [
            (aggregator, ) for aggregator in [
                ymir.garrison.contra, ymir.garrison.fedavg, ymir.garrison.flame, ymir.garrison.foolsgold,
                ymir.garrison.krum, ymir.garrison.median, ymir.garrison.norm_clipping, ymir.garrison.phocas,
                ymir.garrison.trmean, ymir.garrison.viceroy 
            ]
        ]
    )
    def test_step(self, aggregator):
        captain = aggregator.Captain(Model(), Network())
        weights_before = captain.model.get_weights()
        mean_losses = captain.step()
        self.assertEqual(mean_losses, np.mean([i for i in range(10)]))
        weights_after = captain.model.get_weights()
        for w_before, w_after in zip(weights_before,  weights_after):
            self.assertNotAllEqual(w_before, w_after)

if __name__ == '__main__':
    unittest.main()