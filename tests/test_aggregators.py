import unittest
from parameterized import parameterized

import numpy as np

import chex
import optax
import jax
import jax.numpy as jnp

import ymir

@chex.dataclass
class Client:
    batch_size: int
    epochs: int

class Network:
    def __init__(self, clients):
        self.clients = clients

    def __len__(self):
        return len(self.clients)

@chex.dataclass
class Params:
    w: chex.ArrayDevice
    b: chex.ArrayDevice


class TestAggregators(unittest.TestCase):
    def setUp(self):
        self.params = Params(w=jnp.ones(10, dtype=jnp.float32), b=jnp.ones(2, dtype=jnp.float32))
        self.network = Network([Client(batch_size=32, epochs=10) for _ in range(10)])
        self.opt = optax.sgd(0.1)
        self.opt_state = self.opt.init(self.params)
        self.rng = np.random.default_rng()

    @parameterized.expand([
            (server_name) for server_name in ["fedavg", "foolsgold", "krum", "norm_clipping", "std_dagmm", "viceroy"]
    ])
    def test_scale_servers(self, server_name):
        server = getattr(ymir.garrison, server_name).Captain(self.params, self.opt, self.opt_state, self.network, self.rng)
        rngs = jax.random.split(jax.random.PRNGKey(0))
        all_grads = [
            Params(
                w=jax.random.uniform(rngs[0], (10,), dtype=jnp.float32),
                b=jax.random.uniform(rngs[1], (2,), dtype=jnp.float32),
            )
            for _ in self.network.clients
        ]
        server.update(all_grads)
        alpha = server.scale(all_grads)
        chex.assert_tree_no_nones(alpha)
        if server_name != "std_dagmm":
            chex.assert_tree_all_finite(alpha)
        chex.assert_shape(alpha, (len(self.network.clients),))
        chex.assert_type(alpha, jnp.float32)

    @parameterized.expand([
        (server_name) for server_name in ["flguard"]
    ])
    def test_aggregate_servers(self, server_name):
        server = getattr(ymir.garrison, server_name).Captain(self.params, self.opt, self.opt_state, self.network, self.rng)
        rngs = jax.random.split(jax.random.PRNGKey(0))
        all_weights = [
            Params(
                w=jax.random.uniform(rngs[0], (10,), dtype=jnp.float32),
                b=jax.random.uniform(rngs[1], (2,), dtype=jnp.float32),
            )
            for _ in self.network.clients
        ]
        update = server.update(all_weights)
        chex.assert_trees_all_equal_shapes(update, self.params)
        chex.assert_tree_all_finite(update)
        chex.assert_trees_all_equal_dtypes(update, self.params)

        

if __name__ == '__main__':
    unittest.main()