import unittest

import chex
import numpy as np
from parameterized import parameterized

import tfymir


class TestDistributions(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.X = self.rng.random((50, 1))
        self.y = np.sin(self.X).round().reshape(-1)

    @parameterized.expand([(nclients, ) for nclients in range(1, 10)])
    def test_homogeneous(self, nclients):
        dist = tfymir.mp.distributions.homogeneous(self.X, self.y, nclients, 2, self.rng)
        self.assertEqual(len(dist), nclients)
        for d in dist:
            chex.assert_trees_all_close(np.unique(self.y[d]), np.unique(self.y))

    @parameterized.expand([(nclients, ) for nclients in range(1, 10)])
    def test_extreme_heterogeneous(self, nclients):
        dist = tfymir.mp.distributions.extreme_heterogeneous(self.X, self.y, nclients, 2, self.rng)
        self.assertEqual(len(dist), nclients)
        for i, d in enumerate(dist):
            self.assertEqual(np.unique(self.y[d]), i % 2)

    @parameterized.expand([(nclients, ) for nclients in range(1, 5)])
    def test_lda(self, nclients):
        dist = tfymir.mp.distributions.lda(self.X, self.y, nclients, 2, self.rng)
        self.assertEqual(len(dist), nclients)
        for d in dist:
            chex.assert_trees_all_close(np.unique(self.y[d]), np.unique(self.y))

    def test_iid_partition(self):
        dist = tfymir.mp.distributions.iid_partition(self.X, self.y, 2, 2, self.rng)
        self.assertEqual(len(dist), 2)
        for d in dist:
            chex.assert_trees_all_close(np.unique(self.y[d]), np.unique(self.y))

    def test_shard(self):
        dist = tfymir.mp.distributions.shard(self.X, self.y, 2, 2, self.rng)
        self.assertEqual(len(dist), 2)
        for d in dist:
            chex.assert_trees_all_close(np.unique(self.y[d]), np.unique(self.y))

    def test_assign_classes(self):
        dist = tfymir.mp.distributions.assign_classes(self.X, self.y, 2, 2, self.rng, classes=[0, 1])
        self.assertEqual(len(dist), 2)
        for i, d in enumerate(dist):
            chex.assert_trees_all_close(np.unique(self.y[d]), i)


if __name__ == '__main__':
    unittest.main()
