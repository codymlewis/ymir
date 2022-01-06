import itertools

import unittest
from parameterized import parameterized

import jax
import jax.numpy as jnp
import chex

import ymir.lib


@chex.dataclass
class Params:
    """
    Parameter trees for testing.
    """
    w: chex.ArrayDevice
    b: chex.ArrayDevice


class TestTreeFunctions(unittest.TestCase):
    def setUp(self):
        self.length = 50
        self.params = Params(w=jnp.ones(self.length), b=jnp.ones(self.length))

    def test_chain(self):
        f = lambda x: x - 1
        g = lambda x: x**3
        self.assertEqual(ymir.lib.chain([f, g], 2), g(f(2)))

    @parameterized.expand([
        (low, high) for low, high in [(0.5, 0.7), (-0.5, 0.5), (10, 1000), (-1000, -10)]
    ])
    def test_tree_uniform(self, low, high):
        uniform_params = ymir.lib.tree_uniform(self.params, low=low, high=high)
        chex.assert_trees_all_equal_shapes(self.params, uniform_params)
        chex.assert_tree_all_finite(uniform_params)
        chex.assert_tree_no_nones(uniform_params)
        # check that the generated tree values are within the range
        jax.tree_map(lambda x: self.assertTrue((x >= low).all() and (x < high).all()), uniform_params)

    @parameterized.expand([
        (loc, scale) for loc, scale in [(0.0, 1.0), (0.5, 0.7), (-0.5, 0.5), (2, 10), (-1000, 10)]
    ])
    def test_tree_add_normal(self, loc, scale):
        normal_params = ymir.lib.tree_add_normal(self.params, loc=loc, scale=scale)
        chex.assert_trees_all_equal_shapes(self.params, normal_params)
        chex.assert_tree_all_finite(normal_params)
        chex.assert_tree_no_nones(normal_params)
        # check that the generated tree values are within the range
        jax.tree_map(lambda x: self.assertAlmostEqual(x.mean(), loc + 1, delta=scale*0.5), normal_params)
        jax.tree_map(lambda x: self.assertAlmostEqual(x.std(), scale, delta=scale*0.5), normal_params)
    
    @parameterized.expand([
        (mul,) for mul in [2, -1, 0, 1, 3, 5]
    ])
    def test_tree_mul(self, mul):
        mul_params = ymir.lib.tree_mul(self.params, mul)
        chex.assert_trees_all_equal_shapes(self.params, mul_params)
        chex.assert_tree_all_finite(mul_params)
        chex.assert_tree_no_nones(mul_params)
        # check that the generated tree values are within the range
        chex.assert_trees_all_equal_comparator(
            lambda x, y: (x == y*mul).all(), lambda x, y: f"{x} is not a {mul} multiple of {y}", mul_params, self.params
        )

    @parameterized.expand([
            (a, b) for a, b in itertools.product([2, -1, 0, 1, 3, 5], repeat=2)
    ])
    def test_tree_add(self, a, b):
        add_params = ymir.lib.tree_add(
            Params(w=jnp.full(self.length, a), b=jnp.full(self.length, a)),
            Params(w=jnp.full(self.length, b), b=jnp.full(self.length, b))
        )
        chex.assert_trees_all_equal_shapes(self.params, add_params)
        chex.assert_tree_all_finite(add_params)
        chex.assert_tree_no_nones(add_params)
        chex.assert_trees_all_close(add_params, Params(w=jnp.full(self.length, a + b), b=jnp.full(self.length, a + b)))


if __name__ == '__main__':
    unittest.main()