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

class Client:
    def step(self, weights, return_weights=False):
        return 0.0, weights, 2

    def analytics(self):
        return [0.3, 0.4]

class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.network = ymir.mp.network.Network()
        self.clients = [Client() for _ in range(10)]

    def test_member_variables(self):
        self.assertEqual(self.network.clients, [])
        self.assertEqual(self.network.C, 1.0)
        self.assertEqual(self.network.K, 0)
        self.assertEqual(self.network.update_transform_chain, [])
    
    def test_len(self):
        self.assertEqual(len(self.network), 0)
        self.network.add_client(self.clients[0])
        self.assertEqual(len(self.network), 1)
        self.network = ymir.mp.network.Network()

    def test_add_client(self):
        self.network.add_client(self.clients[0])
        self.assertEqual(self.network.clients[0], self.clients[0])
        self.network = ymir.mp.network.Network()

    def test_add_update_transform(self):
        f = lambda x: x
        self.network.add_update_transform(f)
        self.assertEqual(self.network.update_transform_chain[0], f)

    def test_call(self):
        for client in self.clients:
            self.network.add_client(client)
        losses, weights, data = self.network(1.0)
        self.assertEqual(losses, (0.0,) * len(self.clients))
        self.assertEqual(weights, (1.0,) * len(self.clients))
        self.assertEqual(data, (2,) * len(self.clients))
    
    def test_analytics(self):
        for client in self.clients:
            self.network.add_client(client)
        np.testing.assert_array_equal(self.network.analytics(), [[0.3, 0.4]] * len(self.clients))


if __name__ == '__main__':
    unittest.main()