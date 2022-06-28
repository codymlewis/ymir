import unittest

import numpy as np
from parameterized import parameterized
import chex
import jax.numpy as jnp

import ymir


@chex.dataclass
class Params:
    """
    Parameter trees for testing.
    """
    w: chex.ArrayDevice
    b: chex.ArrayDevice


class Network:
    def __init__(self):
        self.grads = []
        self.length = 10

    def __call__(self, weights, rng=np.random.default_rng(), return_weights=False):
        updates = jnp.array([rng.random(weights.shape) for _ in range(self.length)])
        self.grads = updates
        return updates, jnp.array([i for i in range(self.length)]), [32 for _ in range(self.length)]
    
    def __len__(self):
        return self.length


class TestServer(unittest.TestCase):
    @unittest.mock.patch.object(ymir.server.Server, '__abstractmethods__', set())
    def test_member_variables(self):
        network = Network()
        rng = np.random.default_rng()
        params = Params(w=jnp.array([1, 1]), b=jnp.array([1]))
        server = ymir.server.server.Server(network, params, rng=rng)
        self.assertEqual(server.network, network)
        self.assertEqual(server.rng, rng)
        chex.assert_trees_all_equal(server.params, params)


class TestAverage(unittest.TestCase):
    def test_step(self):
        params = jnp.ones(10)
        server = ymir.server.average.Server(Network(), params)
        mean_losses = server.step()
        self.assertEqual(mean_losses, np.mean([i for i in range(10)]))
        chex.assert_tree_all_close(
            params - 1 / len(server.network.grads) * sum(server.network.grads),
            server.params
        )


class TestAggregator(unittest.TestCase):
    @parameterized.expand(
        [
            (aggregator, ) for aggregator in [
                ymir.server.contra, ymir.server.fedavg, ymir.server.flame, ymir.server.foolsgold,
                ymir.server.krum, ymir.server.median, ymir.server.norm_clipping, ymir.server.phocas,
                ymir.server.trmean, 
            ]
        ]
    )
    def test_step(self, aggregator):
        params = jnp.ones(10)
        server = aggregator.Server(Network(), params)
        mean_losses = server.step()
        self.assertEqual(mean_losses, np.mean([i for i in range(10)]))
        chex.assert_trees_all_equal_shapes(params, server.params)
        chex.assert_tree_no_nones(server.params)

if __name__ == '__main__':
    unittest.main()
