import unittest

import chex
import numpy as np

import tfymir


class TestDatasets(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.X = self.rng.random((50, 1))
        self.y = np.sin(self.X).round().reshape(-1)

    def test_DataIter(self):
        dataiter = tfymir.mp.datasets.DataIter(self.X, self.y, 8, 2, self.rng)
        chex.assert_trees_all_close(dataiter.X, self.X)
        chex.assert_trees_all_close(dataiter.y, self.y)
        self.assertEqual(dataiter.batch_size, 8)
        self.assertEqual(dataiter.classes, 2)
        chex.assert_shape(dataiter.idx, (len(self.X), ))
        dataiter = tfymir.mp.datasets.DataIter(self.X, self.y, 500, 2, self.rng)
        self.assertEqual(dataiter.batch_size, len(self.X))

    def test_Dataset(self):
        dataset = tfymir.mp.datasets.Dataset(
            self.X, self.y,
            np.concatenate((np.full(int(len(self.X) * 0.5), True), np.full(int(len(self.X) * 0.5), False)))
        )
        chex.assert_trees_all_close(dataset.X, self.X)
        chex.assert_trees_all_close(dataset.y, self.y)
        self.assertEqual(dataset.classes, 2)
        X, y = dataset.train()
        chex.assert_trees_all_close(X, self.X[:int(len(self.X) * 0.5)])
        chex.assert_trees_all_close(y, self.y[:int(len(self.X) * 0.5)])
        X, y = dataset.test()
        chex.assert_trees_all_close(X, self.X[int(len(self.X) * 0.5):])
        chex.assert_trees_all_close(y, self.y[int(len(self.X) * 0.5):])


if __name__ == '__main__':
    unittest.main()
