import itertools
import unittest

import chex
import jax
import jax.numpy as jnp
from parameterized import parameterized

import ymir.path


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
        self.assertEqual(ymir.path.functions.chain([f, g], 2), g(f(2)))

    @parameterized.expand([(low, high) for low, high in [(0.5, 0.7), (-0.5, 0.5), (10, 1000), (-1000, -10)]])
    def test_tree_uniform(self, low, high):
        uniform_params = ymir.path.tree.uniform(self.params, low=low, high=high)
        chex.assert_trees_all_equal_shapes(self.params, uniform_params)
        chex.assert_tree_all_finite(uniform_params)
        chex.assert_tree_no_nones(uniform_params)
        # check that the generated tree values are within the range
        jax.tree_map(lambda x: self.assertTrue((x >= low).all() and (x < high).all()), uniform_params)

    @parameterized.expand([(loc, scale) for loc, scale in [(0.0, 1.0), (0.5, 0.7), (-0.5, 0.5), (2, 10), (-1000, 10)]])
    def test_tree_add_normal(self, loc, scale):
        normal_params = ymir.path.tree.add_normal(self.params, loc=loc, scale=scale)
        chex.assert_trees_all_equal_shapes(self.params, normal_params)
        chex.assert_tree_all_finite(normal_params)
        chex.assert_tree_no_nones(normal_params)
        # check that the generated tree values are within the range
        jax.tree_map(lambda x: self.assertAlmostEqual(x.mean(), loc + 1, delta=scale * 0.5), normal_params)
        jax.tree_map(lambda x: self.assertAlmostEqual(x.std(), scale, delta=scale * 0.5), normal_params)

    @parameterized.expand([(scale, ) for scale in [2, -1, 0, 1, 3, 5]])
    def test_tree_mul(self, scale):
        mul = Params(w=jnp.ones(self.length) * scale, b=jnp.ones(self.length) * scale)
        mul_params = ymir.path.tree.mul(self.params, mul)
        chex.assert_trees_all_equal_shapes(self.params, mul_params)
        chex.assert_tree_all_finite(mul_params)
        chex.assert_tree_no_nones(mul_params)
        # check that the generated tree values are within the range
        chex.assert_trees_all_equal_comparator(
            lambda x, y: (x == y * scale).all(), lambda x, y: f"{x} is not a {scale} multiple of {y}", mul_params,
            self.params
        )

    @parameterized.expand([(scale, ) for scale in [2, -1, 1, 3, 5]])
    def test_tree_div(self, scale):
        div = Params(w=jnp.ones(self.length) * scale, b=jnp.ones(self.length) * scale)
        div_params = ymir.path.tree.div(self.params, div)
        chex.assert_trees_all_equal_shapes(self.params, div_params)
        chex.assert_tree_all_finite(div_params)
        chex.assert_tree_no_nones(div_params)
        # check that the generated tree values are within the range
        chex.assert_trees_all_equal_comparator(
            lambda x, y: (x == y / scale).all(), lambda x, y: f"{x} is not a {scale} division of {y}", div_params,
            self.params
        )

    @parameterized.expand([(scale, ) for scale in [2, -1, 0, 1, 3, 5]])
    def test_tree_scale(self, scale):
        scale_params = ymir.path.tree.scale(self.params, scale)
        chex.assert_trees_all_equal_shapes(self.params, scale_params)
        chex.assert_tree_all_finite(scale_params)
        chex.assert_tree_no_nones(scale_params)
        # check that the generated tree values are within the range
        chex.assert_trees_all_equal_comparator(
            lambda x, y: (x == y * scale).all(), lambda x, y: f"{x} is not a {scale} multiple of {y}", scale_params,
            self.params
        )

    @parameterized.expand([(a, b) for a, b in itertools.product([2, -1, 0, 1, 3, 5], repeat=2)])
    def test_tree_add(self, a, b):
        add_params = ymir.path.tree.add(
            Params(w=jnp.full(self.length, a), b=jnp.full(self.length, a)),
            Params(w=jnp.full(self.length, b), b=jnp.full(self.length, b))
        )
        chex.assert_trees_all_equal_shapes(self.params, add_params)
        chex.assert_tree_all_finite(add_params)
        chex.assert_tree_no_nones(add_params)
        chex.assert_trees_all_close(add_params, Params(w=jnp.full(self.length, a + b), b=jnp.full(self.length, a + b)))


if __name__ == '__main__':
    unittest.main()
