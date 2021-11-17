from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax
import numpy as np
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


class TestAggregators(parameterized.TestCase):
    def setUp(self):
        self.params = Params(w=jnp.ones(10), b=jnp.ones(2))
        self.network = Network([Client(batch_size=32, epochs=10) for _ in range(10)])

    @parameterized.named_parameters(
        [
            {"testcase_name": f"_{server_name=}", "server_name": server_name}
            for server_name in ["fed_avg", "foolsgold", "krum", "norm_clipping", "std_dagmm", "viceroy"]
        ]
    )
    def test_aggregator(self, server_name):
        server = getattr(ymir.garrison.aggregators, server_name).Server(self.params, self.network)
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


if __name__ == '__main__':
    absltest.main()