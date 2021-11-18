from absl.testing import absltest

import chex
import optax
import haiku as hk
import jax
import numpy as np

import ymir

class TestCoordinateFunctions(absltest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        dataset = ymir.mp.datasets.Dataset((X := rng.random((50, 1))), np.sin(X).reshape(-1), np.concatenate((np.full(30, True), np.full(20, False))))
        data = dataset.fed_split([8 for _ in range(10)], ymir.mp.distributions.homogeneous)
        test_eval = dataset.get_iter("test")
        net = hk.without_apply_rng(hk.transform(lambda x: ymir.mp.models.LeNet_300_100(dataset.classes)(x)))
        self.params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])
        opt = optax.sgd(0.1)
        self.opt_state = opt.init(self.params)
        self.network = ymir.mp.network.Network()
        self.network.add_controller("main", is_server=True)
        for d in data:
            self.network.add_host("main", ymir.scout.Collaborator(opt, self.opt_state, ymir.mp.losses.cross_entropy_loss(net, dataset.classes), d, 1))
        self.model = ymir.Coordinate("fed_avg", opt, self.opt_state, self.params, self.network, rng=rng)

    def test_member_variables(self):
        self.assertIsInstance(self.model.server, ymir.garrison.aggregators.fed_avg.Server)
        self.assertEqual(self.model.params, self.params)
        self.assertEqual(self.model.opt_state, self.opt_state)
        self.assertTrue(callable(self.model.server_update))
        self.assertEqual(self.model.network, self.network)
        self.assertEqual(len(self.model.network), 10)
        self.assertIsInstance(self.model.rng, np.random.Generator)

    def test_step(self):
        alpha, all_grads = self.model.step()
        chex.assert_tree_no_nones(alpha)
        self.params, self.opt_state = self.model.server_update(self.params, self.opt_state, ymir.garrison.sum_grads(all_grads))
        chex.assert_trees_all_close(self.model.params, self.params)


if __name__ == '__main__':
    absltest.main()