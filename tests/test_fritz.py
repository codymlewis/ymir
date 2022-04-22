import unittest
from dataclasses import dataclass
import inspect

import numpy as np
import tensorflow as tf

import ymir.fritz


@dataclass
class StrClient:
    data: str
    step: str


@dataclass
class Model:
    weights: list
@dataclass
class Client:
    global_weights: list
    model: Model

class TestAlternatingMinimization(unittest.TestCase):

    def test_convert(self):
        client = StrClient("data", "step")
        ymir.fritz.alternating_minimization.convert(client, 1, 1, "stealth")
        self.assertEqual(client.poison_step, "step")
        self.assertEqual(client.stealth_data, "stealth")
        self.assertEqual(client.poison_data, "data")
        self.assertEqual(client.poison_epochs, 1)
        self.assertEqual(client.stealth_epochs, 1)
        self.assertEqual(inspect.getmodule(client.stealth_step).__name__, "ymir.regiment.scout")
        self.assertEqual(inspect.getmodule(client.step).__name__, "ymir.fritz.alternating_minimization")

    def test_step(self):
        client = unittest.mock.Mock()
        client.model.get_weights.return_value = []
        client.poison_update = unittest.mock.Mock(return_value=(1, [], 1))
        client.stealth_update = unittest.mock.Mock(return_value=(1, [], 1))
        client.poison_epochs, client.stealth_epochs = 1, 1
        client.step = ymir.fritz.alternating_minimization.step.__get__(client)
        client.step([])
        client.poison_update.assert_called_once()
        client.stealth_update.assert_called_once()


class TestBackdoor(unittest.TestCase):
    def test_convert(self):
        client = unittest.mock.Mock()
        client.data.filter.return_value = client.data
        ymir.fritz.backdoor.convert(client, 0, 1, [])
        client.data.filter.assert_called_once()
        client.data.map.assert_called_once()

    def test_backdoor_map(self):
        X, y = np.zeros((3, 5, 5)), np.array([0, 1, 1])
        trigger = np.ones((3, 3))
        tX, ty = ymir.fritz.backdoor.backdoor_map(0, 1, trigger, X, y)
        self.assertTrue(np.all(tX[y == 0, :3, :3] == trigger))
        self.assertTrue(np.all(tX[y == 1, :3, :3] == 0))
        self.assertTrue(np.all(ty[y == 0] == 1))


class TestConstrainAndScale(unittest.TestCase):

    def test_convert(self):
        client = unittest.mock.Mock()
        client.model.get_weights.return_value = "weights"
        ymir.fritz.constrain_and_scale.convert(client, 1, "distance")
        self.assertEqual(client.alpha, 1)
        self.assertEqual(client.global_weights, "weights")
        self.assertEqual(inspect.getmodule(client.step).__name__, "ymir.fritz.constrain_and_scale")
        self.assertEqual(inspect.getmodule(client._step).__name__, "ymir.fritz.constrain_and_scale")

    def test_step(self):
        client = unittest.mock.Mock()
        client.epochs = 1
        client.data = iter([(0, 0)])
        client.model.get_weights.return_value = "weights"
        client.step = ymir.fritz.constrain_and_scale.step.__get__(client)
        client.step([])
        client.model.get_weights.assert_called_once()
        client.compute_penalty.assert_called_once()
        client._step.assert_called_once()
        client.model.set_weights.assert_called_once()
        self.assertEqual(client.global_weights, [])

    def test_distance_penalty(self):
        client = Client([np.array([0.0, 0.0])], Model([np.array([0.0, 1.0])]))
        self.assertEqual(ymir.fritz.constrain_and_scale._distance_penalty(client), 1)

    def test_cosine_penalty(self):
        client = Client([np.array([1.0, 0.0])], Model([np.array([1.0, 0.0])]))
        self.assertEqual(ymir.fritz.constrain_and_scale._cosine_penalty(client), 2)


class TestFreerider(tf.test.TestCase):
    def test_convert(self):
        client = unittest.mock.Mock()
        client.model.get_weights.return_value = [1]
        ymir.fritz.freerider.convert(client, "type", "rng")
        self.assertEqual(client.attack_type, "type")
        self.assertEqual(client.prev_weights, [1])
        self.assertEqual(client.rng, "rng")
        self.assertEqual(inspect.getmodule(client.step).__name__, "ymir.fritz.freerider")
    
    def test_step(self):
        client = Client([np.array([1.0, 1.0])], Model([np.array([0.0, 0.0])]))
        client.prev_weights = client.global_weights
        client.rng = np.random.default_rng()
        client.batch_size = 1
        client.step = ymir.fritz.freerider.step.__get__(client)
        client.attack_type = "random"
        _, updates, batch_size = client.step([np.ones((1,))])
        self.assertEqual(batch_size, 1)
        self.assertGreaterEqual(updates[0], -10e-3)
        self.assertLessEqual(updates[0], 10e-3)
        client.attack_type = "delta"
        _, updates, _ = client.step([np.array([2.0, 3.0])])
        self.assertAllEqual(updates[0], np.array([1.0, 2.0]))
        client.prev_weights = client.global_weights
        client.attack_type = "advanced delta"
        _, updates, _ = client.step([np.array([2.0, 3.0])])
        self.assertAllClose(updates[0], np.array([1.0, 2.0]), atol=1e-3)


class TestLabelflipper(unittest.TestCase):
    def test_convert(self):
        client = unittest.mock.Mock()
        client.data.filter.return_value = client.data
        ymir.fritz.labelflipper.convert(client, 0, 1)
        client.data.filter.assert_called_once()
        client.data.map.assert_called_once()

    def test_labelflip_map(self):
        X, y = np.zeros((3, 5, 5)), np.array([0, 1, 1])
        tX, ty = ymir.fritz.labelflipper.labelflip_map(0, 1, X, y)
        self.assertTrue(np.all(tX == X))
        self.assertTrue(np.all(ty[y == 0] == 1))


class TestScaler(unittest.TestCase):
    def test_convert(self):
        client = StrClient("data", "step")
        ymir.fritz.scaler.convert(client, 10)
        self.assertEqual(client.quantum_step, "step")

    def test_scale(self):
        loss, updates, batch_size = ymir.fritz.scaler._scale(10, 1, [1], 32)
        self.assertEqual(loss, 1)
        self.assertEqual(updates, [10])
        self.assertEqual(batch_size, 32)


class TestSMP(unittest.TestCase):

    def test_convert(self):
        client = unittest.mock.Mock()
        client.model.get_weights.return_value = "weights"
        ymir.fritz.smp.convert(client, 1, 1, "val")
        self.assertEqual(client.lamb, 1)
        self.assertEqual(client.rho, 1)
        self.assertEqual(client.global_weights, "weights")
        self.assertEqual(inspect.getmodule(client.step).__name__, "ymir.fritz.smp")
        self.assertEqual(inspect.getmodule(client._step).__name__, "ymir.fritz.smp")

    def test_step(self):
        client = Client([np.array([0.0, 0.0])], Model([np.array([0.0, 1.0])]))
        client.model.get_weights = unittest.mock.Mock(return_value=[1.0])
        client.model.set_weights = unittest.mock.Mock()
        client._step = unittest.mock.Mock()
        client.epochs = 1
        client.batch_size = 1
        client.data = iter([(0, 0)])
        client.step = ymir.fritz.smp.step.__get__(client)
        client.step([1])
        client.model.get_weights.assert_called_once()
        client._step.assert_called_once()
        client.model.set_weights.assert_called_once()
        self.assertEqual(client.global_weights, [1])


if __name__ == '__main__':
    unittest.main()