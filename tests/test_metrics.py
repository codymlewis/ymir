import unittest

import chex
import jax.numpy as jnp
import numpy as np

import ymir
from ymir.mp.metrics import Neurometer


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.params = jnp.ones(2)

        class _Net:

            def apply(self, params, X):
                return (params @ X) + jnp.ones(2)

        self.net = _Net()

    def test_evaluator(self):
        y_true = np.array([1])
        evaluator = ymir.mp.metrics.evaluator(self.net)
        y, y_pred = evaluator(self.params, jnp.ones(2), y_true)
        chex.assert_trees_all_close(y, y_true)
        self.assertEqual(y_pred, 0)

    def test_accuracy_score(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        acc = ymir.mp.metrics.accuracy_score(y_true, y_pred)
        self.assertEqual(acc, 0.5)

    def test_neurometer(self):

        class _Dataset:

            def __init__(self, X):
                self.X = X
                self.classes = 2

        datasets = {'train': _Dataset(jnp.ones(2)), 'valid': _Dataset(jnp.ones(2))}
        meter = Neurometer(self.net, datasets)
        self.assertDictEqual(meter.datasets, datasets)
        self.assertDictEqual(meter.results, {d: [] for d in datasets.keys()})
        self.assertDictEqual(meter.classes, {d: ds.classes for d, ds in datasets.items()})


if __name__ == '__main__':
    unittest.main()
