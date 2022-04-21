import itertools
import unittest
import collections

import numpy as np
import tensorflow as tf
from parameterized import parameterized

import ymir.path


class TestFunctions(unittest.TestCase):
    def test_chain(self):
        f = lambda x: x - 1
        g = lambda x: x**3
        self.assertEqual(ymir.path.functions.chain([f, g], 2), g(f(2)))

class TestWeights(tf.test.TestCase):

    def setUp(self):
        self.length = 50
        self.params = [np.ones(self.length) for _ in range(5)]

    @parameterized.expand([(low, high) for low, high in [(0.5, 0.7), (-0.5, 0.5), (10, 1000), (-1000, -10)]])
    def test_uniform(self, low, high):
        uniform_params = ymir.path.weights.uniform(self.params, low=low, high=high)
        for up, p in zip(uniform_params, self.params):
            self.assertShapeEqual(up, p)
            self.assertAllGreaterEqual(up, low)
            self.assertAllLessEqual(up, high)

    @parameterized.expand([(loc, scale) for loc, scale in [(0.0, 1.0), (0.5, 0.7), (-0.5, 0.5), (2, 10), (-1000, 10)]])
    def test_add_normal(self, loc, scale):
        normal_params = ymir.path.weights.add_normal(self.params, loc=loc, scale=scale)
        for gp, p in zip(normal_params, self.params):
            self.assertShapeEqual(gp, p)
            self.assertAllClose(gp.mean(), loc + 1, atol=scale * 0.5)
            self.assertAllClose(gp.std(), scale, atol=scale * 0.5)

    @parameterized.expand([(scale, ) for scale in [2, -1, 0, 1, 3, 5]])
    def test_mul(self, scale):
        mul = [np.ones_like(p) * scale for p in self.params]
        mul_params = ymir.path.weights.mul(self.params, mul)
        for mp, p in zip(mul_params, self.params):
            self.assertShapeEqual(mp, p)
            self.assertAllEqual(mp, p * scale)

    @parameterized.expand([(scale, ) for scale in [2, -1, 1, 3, 5]])
    def test_div(self, scale):
        div = [np.ones_like(p) * scale for p in self.params]
        div_params = ymir.path.weights.div(self.params, div)
        for mp, p in zip(div_params, self.params):
            self.assertShapeEqual(mp, p)
            self.assertAllEqual(mp, p / scale)

    @parameterized.expand([(scale, ) for scale in [2, -1, 0, 1, 3, 5]])
    def test_scale(self, scale):
        scale_params = ymir.path.weights.scale(self.params, scale)
        for sp, p in zip(scale_params, self.params):
            self.assertShapeEqual(sp, p)
            self.assertAllEqual(sp, p * scale)


    @parameterized.expand([(a, b) for a, b in itertools.product([2, -1, 0, 1, 3, 5], repeat=2)])
    def test_add(self, a, b):
        add_params = ymir.path.weights.add(
            [np.full(self.length, a) for _ in range(5)],
            [np.full(self.length, b) for _ in range(5)]
        )
        expected_row = np.full(self.length, a + b)
        for ap in add_params:
            self.assertShapeEqual(ap, expected_row)
            self.assertAllEqual(ap, expected_row)

    @parameterized.expand([(a, b) for a, b in itertools.product([2, -1, 0, 1, 3, 5], repeat=2)])
    def test_sub(self, a, b):
        sub_params = ymir.path.weights.sub(
            [np.full(self.length, a) for _ in range(5)],
            [np.full(self.length, b) for _ in range(5)]
        )
        expected_row = np.full(self.length, a - b)
        for sp in sub_params:
            self.assertShapeEqual(sp, expected_row)
            self.assertAllEqual(sp, expected_row)

    def test_ravel(self):
        params = [np.ones(self.length) for _ in range(5)]
        ravel_params = ymir.path.weights.ravel(params)
        self.assertAllEqual(np.ones(self.length * 5), ravel_params)

    def test_unravel(self):
        params = [np.ones(self.length) for _ in range(5)]
        ravel_params = ymir.path.weights.ravel(params)
        skeleton = ymir.path.weights.unraveller(params)
        unravelled_params = ymir.path.weights.unravel(ravel_params, skeleton)
        for up, p in zip(unravelled_params, params):
            self.assertShapeEqual(up, p)
            self.assertAllEqual(up, p)

    def test_unraveller(self):
        params = [np.ones((self.length, self.length)) for _ in range(5)]
        unraveller = ymir.path.weights.unraveller(params)
        for p, (shape, length) in zip(params, unraveller):
            self.assertEqual(p.shape, shape)
            self.assertEqual(p.size, length)

    def test_get_names(self):
        Weight = collections.namedtuple('Weight', ['name'])
        model = unittest.mock.Mock(weights=[Weight(name=f"w{i}") for i in range(5)])
        names = ymir.path.weights.get_names(model)
        self.assertEqual(names, [f"w{i}" for i in range(5)])

    @parameterized.expand([(val, ) for val in [2, -1, 0, 1, 3, 5]])
    def test_minimum(self, val):
        rng = np.random.default_rng()
        params = [np.array(rng.integers(-10, 10, self.length)) for _ in range(5)]
        minimum_params = ymir.path.weights.minimum(params, val)
        for p, mp in zip(params, minimum_params):
            self.assertShapeEqual(p, mp)
            self.assertAllEqual(np.minimum(p, val), mp)

    @parameterized.expand([(val, ) for val in [2, -1, 0, 1, 3, 5]])
    def test_maximum(self, val):
        rng = np.random.default_rng()
        params = [np.array(rng.integers(-10, 10, self.length)) for _ in range(5)]
        maximum_params = ymir.path.weights.maximum(params, val)
        for p, mp in zip(params, maximum_params):
            self.assertShapeEqual(p, mp)
            self.assertAllEqual(np.maximum(p, val), mp)


if __name__ == '__main__':
    unittest.main()
