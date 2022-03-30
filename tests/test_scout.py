import unittest

import haiku as hk
import hkzoo
import jax
import numpy as np
import optax

import tfymir


class TestScoutFunctions(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        dataset = tfymir.mp.datasets.Dataset(
            (X := rng.random((50, 1), dtype=np.float32)),
            np.sin(X).reshape(-1), np.full(len(X), True)
        )
        self.data = dataset.get_iter("train")
        net = hk.without_apply_rng(hk.transform(lambda x: hkzoo.LeNet_300_100(dataset.classes, x)))
        params = net.init(jax.random.PRNGKey(42), next(self.data)[0])
        opt = optax.sgd(0.1)
        self.opt_state = opt.init(params)
        self.client = tfymir.regiment.Scout(None, self.opt_state, None, self.data, 1)

    def test_member_variables(self):
        self.assertEqual(self.client.opt_state, self.opt_state)
        self.assertEqual(self.client.data, self.data)
        self.assertEqual(self.client.batch_size, 50)
        self.assertEqual(self.client.epochs, 1)
        self.assertTrue(callable(self.client.update))


if __name__ == '__main__':
    unittest.main()
