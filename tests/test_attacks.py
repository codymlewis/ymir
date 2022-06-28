import unittest
from dataclasses import dataclass
import inspect

import numpy as np
import jax.numpy as jnp
import chex

import ymir


@dataclass
class StrClient:
    data: str
    step: str
    opt: str
    loss: str


@dataclass
class Model:
    params: chex.Array


@dataclass
class Client:
    params: chex.Array
    model: Model


class TestAlternatingMinimization(unittest.TestCase):

    def test_convert(self):
        client = StrClient("data", "step", "opt", "loss")
        ymir.attacks.alternating_minimization.convert(client, 1, 1, "stealth")
        self.assertEqual(client.poison_step, "step")
        self.assertEqual(client.stealth_data, "stealth")
        self.assertEqual(client.poison_epochs, 1)
        self.assertEqual(client.stealth_epochs, 1)
        self.assertEqual(inspect.getmodule(client.stealth_step).__name__, "ymir.client.client")
        self.assertEqual(inspect.getmodule(client.step).__name__, "ymir.attacks.alternating_minimization")

    def test_step(self):
        client = unittest.mock.Mock()
        client.model.get_weights.return_value = []
        client.poison_step = unittest.mock.Mock(return_value=([], 1))
        client.stealth_step = unittest.mock.Mock(return_value=([], 1))
        client.poison_epochs, client.stealth_epochs = 1, 1
        client.stealth_data = iter([(1, 1)])
        client.step = ymir.attacks.alternating_minimization.step.__get__(client)
        client.step([], [], [], [])
        client.poison_step.assert_called_once()
        client.stealth_step.assert_called_once()


class TestBackdoor(unittest.TestCase):
    def test_convert(self):
        client = unittest.mock.Mock()
        client.data.filter.return_value = client.data
        ymir.attacks.backdoor.convert(client, 0, 1, [])
        client.data.filter.assert_called_once()
        client.data.map.assert_called_once()

    def test_backdoor_map(self):
        X, y = np.zeros((3, 5, 5)), np.array([0, 1, 1])
        trigger = np.ones((3, 3))
        tX, ty = ymir.attacks.backdoor.backdoor_map(0, 1, trigger, X, y)
        self.assertTrue(np.all(tX[y == 0, :3, :3] == trigger))
        self.assertTrue(np.all(tX[y == 1, :3, :3] == 0))
        self.assertTrue(np.all(ty[y == 0] == 1))


class TestFreerider(unittest.TestCase):
    def test_convert(self):
        client = unittest.mock.Mock()
        client.params = jnp.array([1])
        ymir.attacks.freerider.convert(client, "type", "rng")
        self.assertEqual(client.attack_type, "type")
        chex.assert_trees_all_equal(client.prev_params, jnp.array([1]))
        self.assertEqual(client.rng, "rng")
        self.assertEqual(inspect.getmodule(client.step).__name__, "ymir.attacks.freerider")
    
    def test_step(self):
        client = Client(jnp.array([1.0, 1.0]), Model(jnp.array([0.0, 0.0])))
        client.prev_params = client.params
        client.rng = np.random.default_rng()
        client.batch_size = 1
        client.step = ymir.attacks.freerider.step.__get__(client)
        client.attack_type = "random"
        updates, _, batch_size = client.step(jnp.ones((2,)))
        self.assertEqual(batch_size, 1)
        self.assertGreaterEqual(updates[0], -10e-3)
        self.assertLessEqual(updates[0], 10e-3)
        client.attack_type = "delta"
        updates, _, _ = client.step(jnp.array([2.0, 3.0]))
        chex.assert_trees_all_equal(updates, jnp.array([1.0, 2.0]))
        client.prev_params = client.params
        client.attack_type = "advanced delta"
        updates, _, _ = client.step(jnp.array([2.0, 3.0]))
        chex.assert_trees_all_close(updates, jnp.array([1.0, 2.0]), atol=1e-3)


class TestLabelflipper(unittest.TestCase):
    def test_convert(self):
        client = unittest.mock.Mock()
        client.data.filter.return_value = client.data
        ymir.attacks.label_flipper.convert(client, 0, 1)
        client.data.filter.assert_called_once()
        client.data.map.assert_called_once()

    def test_labelflip_map(self):
        X, y = np.zeros((3, 5, 5)), np.array([0, 1, 1])
        tX, ty = ymir.attacks.label_flipper.labelflip_map(0, 1, X, y)
        self.assertTrue(np.all(tX == X))
        self.assertTrue(np.all(ty[y == 0] == 1))


class TestScaler(unittest.TestCase):
    def test_convert(self):
        client = StrClient("data", "step", "opt", "loss")
        ymir.attacks.scaler.convert(client, 10)
        self.assertEqual(client.quantum_step, "step")

    def test_scale(self):
        updates, loss, batch_size = ymir.attacks.scaler._scale(10, 1, jnp.array([1]), 32)
        self.assertEqual(loss, 1)
        chex.assert_trees_all_equal(updates, jnp.array([10]))
        self.assertEqual(batch_size, 32)


if __name__ == '__main__':
    unittest.main()
