from absl.testing import absltest
from absl.testing import parameterized

import optax
import haiku as hk
import jax
import numpy as np

import ymir

# class TestCollaboratorFunctions(parameterized.TestCase):
#     def setUp(self):
#         rng = np.random.default_rng(0)
#         dataset = ymir.mp.datasets.Dataset((X := rng.random((50, 1))), np.sin(X).reshape(-1), np.full(len(X), True))
#         self.data = dataset.get_iter("train")
#         net = hk.without_apply_rng(hk.transform(lambda x: ymir.mp.models.LeNet_300_100(dataset.classes)(x)))
#         params = net.init(jax.random.PRNGKey(42), next(self.data)[0])
#         opt = optax.sgd(0.1)
#         self.opt_state = opt.init(params)
#         self.client = ymir.scout.Collaborator(self.opt_state, self.data, 1)

#     def test_member_variables(self):
#         self.assertEqual(self.client.opt_state, self.opt_state)
#         self.assertEqual(self.client.data, self.data)
#         self.assertEqual(self.client.batch_size, 50)
#         self.assertEqual(self.client.epochs, 1)


if __name__ == '__main__':
    absltest.main()