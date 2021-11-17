from absl.testing import absltest
from absl.testing import parameterized

import chex
import numpy as np

import ymir


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
        for d in dist:
            chex.assert_trees_all_close(np.unique(self.y[d]), np.unique(self.y))


if __name__ == '__main__':
    absltest.main()