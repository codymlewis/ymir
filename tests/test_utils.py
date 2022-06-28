import unittest

import numpy as np
from parameterized import parameterized
import jax.numpy as jnp
import chex

import ymir


class TestDistributions(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.X = self.rng.random((50, 1))
        self.y = np.sin(self.X).round().reshape(-1)

    @parameterized.expand([(nclients, ) for nclients in range(1, 10)])
    def test_homogeneous(self, nclients):
        dist = ymir.utils.distributions.homogeneous(self.y, nclients, 2, self.rng)
        self.assertEqual(len(dist), nclients)
        for d in dist:
            np.testing.assert_allclose(np.unique(self.y[d]), np.unique(self.y))

    @parameterized.expand([(nclients, ) for nclients in range(1, 10)])
    def test_extreme_heterogeneous(self, nclients):
        dist = ymir.utils.distributions.extreme_heterogeneous(self.y, nclients, 2, self.rng)
        self.assertEqual(len(dist), nclients)
        for i, d in enumerate(dist):
            self.assertEqual(np.unique(self.y[d]), i % 2)

    @parameterized.expand([(nclients, ) for nclients in range(1, 5)])
    def test_lda(self, nclients):
        dist = ymir.utils.distributions.lda(self.y, nclients, 2, self.rng)
        self.assertEqual(len(dist), nclients)
        for d in dist:
            np.testing.assert_allclose(np.unique(self.y[d]), np.unique(self.y))

    def test_iid_partition(self):
        dist = ymir.utils.distributions.iid_partition(self.y, 2, 2, self.rng)
        self.assertEqual(len(dist), 2)
        for d in dist:
            np.testing.assert_allclose(np.unique(self.y[d]), np.unique(self.y))

    def test_shard(self):
        dist = ymir.utils.distributions.shard(self.y, 2, 2, self.rng)
        self.assertEqual(len(dist), 2)
        for d in dist:
            np.testing.assert_allclose(np.unique(self.y[d]), np.unique(self.y))

    def test_assign_classes(self):
        dist = ymir.utils.distributions.assign_classes(self.y, 2, 2, self.rng, classes=[0, 1])
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
        self.network = ymir.utils.network.Network()
        self.clients = [Client() for _ in range(10)]

    def test_member_variables(self):
        self.assertEqual(self.network.clients, [])
        self.assertEqual(self.network.C, 1.0)
        self.assertEqual(self.network.K, 0)

    def test_len(self):
        self.assertEqual(len(self.network), 0)
        self.network.add_client(self.clients[0])
        self.assertEqual(len(self.network), 1)
        self.network = ymir.utils.network.Network()

    def test_add_client(self):
        self.network.add_client(self.clients[0])
        self.assertEqual(self.network.clients[0], self.clients[0])
        self.network = ymir.utils.network.Network()

    def test_call(self):
        for client in self.clients:
            self.network.add_client(client)
        losses, weights, data = self.network(1.0)
        np.testing.assert_array_equal(losses, np.repeat(0.0, len(self.clients)))
        np.testing.assert_array_equal(weights, np.repeat(1.0, len(self.clients)))
        np.testing.assert_array_equal(data, np.repeat(2, len(self.clients)))

    def test_analytics(self):
        for client in self.clients:
            self.network.add_client(client)
        np.testing.assert_array_equal(self.network.analytics(), [[0.3, 0.4]] * len(self.clients))


@chex.dataclass
class Params:
    """
    Parameter trees for testing.
    """
    w: chex.ArrayDevice
    b: chex.ArrayDevice


class TestFunctions(unittest.TestCase):
    def test_ravel(self):
        ravelled_params = ymir.utils.functions.ravel(Params(w=jnp.array([1, 1]), b=jnp.array([1])))
        np.testing.assert_array_equal(jnp.array([1, 1, 1]), ravelled_params)
    
    def test_gradient(self):
        a = Params(w=jnp.array([3, 3]), b=jnp.array([3]))
        b = Params(w=jnp.array([1, 2]), b=jnp.array([3]))
        grad = ymir.utils.functions.gradient(a, b)
        np.testing.assert_array_equal(np.array([0, 2, 1]), grad)

    def test_scale_sum(self):
        ss = ymir.utils.functions.scale_sum(jnp.array([[1, 1], [0, 0], [2, 3]]), jnp.array([0.1, 3, 5]))
        np.testing.assert_array_equal(jnp.array([10.1, 15.1], dtype=jnp.float32), ss)


if __name__ == '__main__':
    unittest.main()
