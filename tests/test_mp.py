import unittest

import numpy as np
from parameterized import parameterized

import ymir


class TestDatasets(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.X = self.rng.random((50, 1))
        self.y = np.sin(self.X).round().reshape(-1)

    def test_DataIter(self):
        dataiter = ymir.mp.datasets.DataIter(self.X, self.y, 8, 2, self.rng)
        np.testing.assert_allclose(dataiter.X, self.X)
        np.testing.assert_allclose(dataiter.y, self.y)
        self.assertEqual(dataiter.batch_size, 8)
        self.assertEqual(dataiter.classes, 2)
        self.assertEqual(dataiter.idx.shape, (len(self.X), ))
        dataiter = ymir.mp.datasets.DataIter(self.X, self.y, 500, 2, self.rng)
        self.assertEqual(dataiter.batch_size, len(self.X))

    def test_Dataset(self):
        dataset = ymir.mp.datasets.Dataset(
            self.X, self.y,
            np.concatenate((np.full(int(len(self.X) * 0.5), True), np.full(int(len(self.X) * 0.5), False)))
        )
        np.testing.assert_allclose(dataset.X, self.X)
        np.testing.assert_allclose(dataset.y, self.y)
        self.assertEqual(dataset.classes, 2)
        X, y = dataset.train()
        np.testing.assert_allclose(X, self.X[:int(len(self.X) * 0.5)])
        np.testing.assert_allclose(y, self.y[:int(len(self.X) * 0.5)])
        X, y = dataset.test()
        np.testing.assert_allclose(X, self.X[int(len(self.X) * 0.5):])
        np.testing.assert_allclose(y, self.y[int(len(self.X) * 0.5):])


class TestDistributions(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.X = self.rng.random((50, 1))
        self.y = np.sin(self.X).round().reshape(-1)

    @parameterized.expand([(nclients, ) for nclients in range(1, 10)])
    def test_homogeneous(self, nclients):
        dist = ymir.mp.distributions.homogeneous(self.X, self.y, nclients, 2, self.rng)
        self.assertEqual(len(dist), nclients)
        for d in dist:
            np.testing.assert_allclose(np.unique(self.y[d]), np.unique(self.y))

    @parameterized.expand([(nclients, ) for nclients in range(1, 10)])
    def test_extreme_heterogeneous(self, nclients):
        dist = ymir.mp.distributions.extreme_heterogeneous(self.X, self.y, nclients, 2, self.rng)
        self.assertEqual(len(dist), nclients)
        for i, d in enumerate(dist):
            self.assertEqual(np.unique(self.y[d]), i % 2)

    @parameterized.expand([(nclients, ) for nclients in range(1, 5)])
    def test_lda(self, nclients):
        dist = ymir.mp.distributions.lda(self.X, self.y, nclients, 2, self.rng)
        self.assertEqual(len(dist), nclients)
        for d in dist:
            np.testing.assert_allclose(np.unique(self.y[d]), np.unique(self.y))

    def test_iid_partition(self):
        dist = ymir.mp.distributions.iid_partition(self.X, self.y, 2, 2, self.rng)
        self.assertEqual(len(dist), 2)
        for d in dist:
            np.testing.assert_allclose(np.unique(self.y[d]), np.unique(self.y))

    def test_shard(self):
        dist = ymir.mp.distributions.shard(self.X, self.y, 2, 2, self.rng)
        self.assertEqual(len(dist), 2)
        for d in dist:
            np.testing.assert_allclose(np.unique(self.y[d]), np.unique(self.y))

    def test_assign_classes(self):
        dist = ymir.mp.distributions.assign_classes(self.X, self.y, 2, 2, self.rng, classes=[0, 1])
        self.assertEqual(len(dist), 2)
        for i, d in enumerate(dist):
            np.testing.assert_allclose(np.unique(self.y[d]), i)

# class TestNetwork(unittest.TestCase):

#     def test_controller(self):
#         controller = ymir.mp.network.Controller(0.1)
#         self.assertListEqual(controller.clients, [])
#         self.assertListEqual(controller.switches, [])
#         self.assertEqual(controller.C, 0.1)
#         self.assertEqual(controller.K, 0)

#     def test_network(self):
#         network = ymir.mp.network.Network(0.1)
#         self.assertListEqual(network.clients, [])
#         self.assertDictEqual(network.controllers, {})
#         self.assertEqual(network.server_name, "")
#         self.assertEqual(network.C, 0.1)

if __name__ == '__main__':
    unittest.main()