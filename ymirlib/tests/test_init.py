import unittest
import itertools

import jax
import jax.numpy as jnp
import chex

import ymirlib


@chex.dataclass
class Params:
    """
    Parameter trees for testing.
    """
    w: chex.ArrayDevice
    b: chex.ArrayDevice


class TestTreeFunctions(unittest.TestCase):
    def test_tree_uniform(self):
        params = Params(w=jnp.zeros(10), b=jnp.zeros(10))
        uniform_params = ymirlib.tree_uniform(params)
        chex.assert_trees_all_equal_shapes(params, uniform_params)
        chex.assert_tree_all_finite(uniform_params)
        chex.assert_tree_no_nones(uniform_params)
        # check that the generated tree values are within the range
        jax.tree_map(lambda x: self.assertTrue((x >= 0.0).all() and (x < 1.0).all()), uniform_params)
        # check for other values
        for (low, high) in [(0.5, 0.7), (-0.5, 0.5), (10, 1000), (-1000, -10)]:
            uniform_params = ymirlib.tree_uniform(params, low=low, high=high)
            jax.tree_map(lambda x: self.assertTrue((x >= low).all() and (x < high).all()), uniform_params)

    def test_tree_normal(self):
        length = 150_000
        params = Params(w=jnp.zeros(length), b=jnp.zeros(length))
        normal_params = ymirlib.tree_add_normal(params)
        chex.assert_trees_all_equal_shapes(params, normal_params)
        chex.assert_tree_all_finite(normal_params)
        chex.assert_tree_no_nones(normal_params)
        # check that the generated tree values are within the range
        jax.tree_map(lambda x: self.assertAlmostEqual(x.mean(), 0.0, 1), normal_params)
        jax.tree_map(lambda x: self.assertAlmostEqual(x.std(), 1.0, 1), normal_params)
        # check for other values
        for (mean, std) in [(0.5, 0.7), (-0.5, 0.5), (2, 10), (-1000, 10)]:
            normal_params = ymirlib.tree_add_normal(params, loc=mean, scale=std)
            jax.tree_map(lambda x: self.assertAlmostEqual(x.mean(), mean, delta=std*0.01), normal_params)
            jax.tree_map(lambda x: self.assertAlmostEqual(x.std(), std, 1), normal_params)
        # check adding properties
        params = Params(w=jnp.ones(length), b=jnp.ones(length))
        for (mean, std) in [(0.0, 1.0), (0.5, 0.7), (-0.5, 0.5), (2, 10), (-1000, 10)]:
            normal_params = ymirlib.tree_add_normal(params, loc=mean, scale=std)
            jax.tree_map(lambda x: self.assertAlmostEqual(x.mean(), mean + 1, delta=std*0.01), normal_params)
            jax.tree_map(lambda x: self.assertAlmostEqual(x.std(), std, 1), normal_params)
    
    def test_tree_mul(self):
        params = Params(w=jnp.zeros(10), b=jnp.zeros(10))
        mul_params = ymirlib.tree_mul(params, 2)
        chex.assert_trees_all_equal_shapes(params, mul_params)
        chex.assert_tree_all_finite(mul_params)
        chex.assert_tree_no_nones(mul_params)
        # check that the generated tree values are within the range
        chex.assert_trees_all_equal_comparator(lambda x, y: (x == y*2).all(), lambda x, y: f"{x} is not a 2 multiple of {y}", params, mul_params)
        # check for other values
        for mul in [-1, 0, 1, 3, 5]:
            mul_params = ymirlib.tree_mul(params, mul)
            chex.assert_trees_all_equal_comparator(lambda x, y: (x == y*mul).all(), lambda x, y: f"{x} is not a {mul} multiple of {y}", params, mul_params)

    def test_tree_add(self):
        length = 10
        params = Params(w=jnp.zeros(length), b=jnp.zeros(length))
        add_params = ymirlib.tree_add(params, Params(w=jnp.full(length, 1), b=jnp.full(length, 1)))
        chex.assert_trees_all_equal_shapes(params, add_params)
        chex.assert_tree_all_finite(add_params)
        chex.assert_tree_no_nones(add_params)
        # check that the generated tree values are within the range
        chex.assert_trees_all_equal_comparator(lambda x, y: (x + 1 == y).all(), lambda x, y: f"{x} is not {y} + 1", params, add_params)
        # check for other values
        for add in [-1, 0, 1, 3, 5]:
            add_params = ymirlib.tree_add(params, Params(w=jnp.full(length, add), b=jnp.full(length, add)))
            chex.assert_trees_all_equal_comparator(lambda x, y: (x + add == y).all(), lambda x, y: f"{x} is not {y} + {add}", params, add_params)
        # check for pairs of values
        for add1, add2 in itertools.product([-1, 0, 1, 3, 5], repeat=2):
            add_params = ymirlib.tree_add(Params(w=jnp.full(length, add1), b=jnp.full(length, add1)), Params(w=jnp.full(length, add2), b=jnp.full(length, add2)))
            jax.tree_map(lambda x: self.assertTrue((x == add1 + add2).all()), add_params)


if __name__ == '__main__':
    unittest.main()