import unittest

import jax.numpy as jnp

import ymir


class TestLosses(unittest.TestCase):
    def setUp(self):
        self.params = jnp.ones(2)
        class _Net:
            def apply(self, params, X):
                return (params @ X) + jnp.ones(2)
        self.net = _Net()

    def test_cross_entropy_loss(self):
        loss = ymir.mp.losses.cross_entropy_loss(self.net, 2)
        self.assertAlmostEqual(loss(self.params, jnp.ones(2), jnp.array([1])), 0.69, places=2)
        self.assertAlmostEqual(loss(self.params, jnp.ones(2), jnp.array([0])), 0.69, places=2)

    def test_ae_l2_loss(self):
        loss = ymir.mp.losses.ae_l2_loss(self.net)
        self.assertAlmostEqual(loss(self.params, jnp.array([1, 1])), 2, places=1)


if __name__ == '__main__':
    unittest.main()