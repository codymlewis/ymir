from absl.testing import absltest
from absl.testing import parameterized

import optax
import chex
import haiku as hk
import jax
import numpy as np

import ymir

class TestDatasets(absltest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.X = self.rng.random((50, 1))
        self.y = np.sin(self.X).round().reshape(-1)

    def test_DataIter(self):
        dataiter = ymir.mp.datasets.DataIter(self.X, self.y, 8, 2, self.rng)
        chex.assert_trees_all_close(dataiter.X, self.X)
        chex.assert_trees_all_close(dataiter.y, self.y)
        self.assertEqual(dataiter.batch_size, 8)
        self.assertEqual(dataiter.classes, 2)
        chex.assert_shape(dataiter.idx, (len(self.X),))
        dataiter = ymir.mp.datasets.DataIter(self.X, self.y, 500, 2, self.rng)
        self.assertEqual(dataiter.batch_size, len(self.X))
    
    def test_Dataset(self):
        dataset = ymir.mp.datasets.Dataset(self.X, self.y, np.concatenate((np.full(int(len(self.X) * 0.5), True), np.full(int(len(self.X) * 0.5), False))))
        chex.assert_trees_all_close(dataset.X, self.X)
        chex.assert_trees_all_close(dataset.y, self.y)
        self.assertEqual(dataset.classes, 2)
        X, y = dataset.train()
        chex.assert_trees_all_close(X, self.X[:int(len(self.X) * 0.5)])
        chex.assert_trees_all_close(y, self.y[:int(len(self.X) * 0.5)])
        X, y = dataset.test()
        chex.assert_trees_all_close(X, self.X[int(len(self.X) * 0.5):])
        chex.assert_trees_all_close(y, self.y[int(len(self.X) * 0.5):])


class TestDistributions(parameterized.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.X = self.rng.random((50, 1))
        self.y = np.sin(self.X).round().reshape(-1)

    @parameterized.named_parameters(
        [
            {"testcase_name": f"_{nendpoints=}", "nendpoints": nendpoints}
            for nendpoints in range(1, 10)
        ]
    )
    def test_homogeneous(self, nendpoints):
        dist = ymir.mp.distributions.homogeneous(self.X, self.y, nendpoints, 2, self.rng)
        self.assertEqual(len(dist), nendpoints)
        for d in dist:
            chex.assert_trees_all_close(np.unique(self.y[d]), np.unique(self.y))

    @parameterized.named_parameters(
        [
            {"testcase_name": f"_{nendpoints=}", "nendpoints": nendpoints}
            for nendpoints in range(1, 10)
        ]
    )
    def test_extreme_heterogeneous(self, nendpoints):
        dist = ymir.mp.distributions.extreme_heterogeneous(self.X, self.y, nendpoints, 2, self.rng)
        self.assertEqual(len(dist), nendpoints)
        for i, d in enumerate(dist):
            self.assertEqual(np.unique(self.y[d]), i % 2)

    @parameterized.named_parameters(
        [
            {"testcase_name": f"_{nendpoints=}", "nendpoints": nendpoints}
            for nendpoints in range(1, 5)
        ]
    )
    def test_lda(self, nendpoints):
        dist = ymir.mp.distributions.lda(self.X, self.y, nendpoints, 2, self.rng)
        self.assertEqual(len(dist), nendpoints)
        for d in dist:
            chex.assert_trees_all_close(np.unique(self.y[d]), np.unique(self.y))

    def test_iid_partition(self):
        dist = ymir.mp.distributions.iid_partition(self.X, self.y, 2, 2, self.rng)
        self.assertEqual(len(dist), 2)
        for d in dist:
            chex.assert_trees_all_close(np.unique(self.y[d]), np.unique(self.y))

    def test_shard(self):
        dist = ymir.mp.distributions.shard(self.X, self.y, 2, 2, self.rng)
        self.assertEqual(len(dist), 2)
        for i, d in enumerate(dist):
            chex.assert_trees_all_close(np.unique(self.y[d]), np.unique(self.y))


if __name__ == '__main__':
    absltest.main()