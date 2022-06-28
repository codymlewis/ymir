import unittest
import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import chex

import ymir

class Net(nn.Module):
    classes: int

    @nn.compact
    def __call__(self, x, act=False):
        x = nn.Dense(3)(x)
        x = nn.relu(x)
        if act:
            return x
        return nn.Dense(self.classes)(x)

def create_model(input_shape, output_shape):
    model = Net(output_shape)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros(input_shape))
    opt = optax.sgd(0.1)
    loss = lambda p, X, y: jnp.mean(optax.softmax_cross_entropy(model.apply(p, X), jax.nn.one_hot(y, output_shape)))
    return params, opt, loss


class Dataset:
    def __init__(self):
        self.X = np.arange(-np.pi, np.pi, 0.1).reshape(-1, 1).astype(np.float32)
        self.y = (np.sin(self.X) > 0).reshape(-1).astype(np.int8)
        self.batch_size = len(self.y)
        self.input_shape = self.X.shape[1:]
        self.classes = np.unique(self.y).shape[0]

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.X, self.y


class TestClient(unittest.TestCase):

    def setUp(self):
        self.dataset = Dataset()
        self.epochs = 1
        self.params, self.opt, self.loss = create_model(self.dataset.input_shape, self.dataset.classes)
        self.client = ymir.client.Client(self.params, self.opt, self.loss, self.dataset, self.epochs)

    def test_member_variables(self):
        self.assertEqual(self.client.data, self.dataset)
        self.assertEqual(self.client.epochs, self.epochs)
        chex.assert_trees_all_equal(self.client.params, self.params)

    def test_step(self):
        updates, loss, batch_size = self.client.step(self.params)
        self.assertEqual(batch_size, self.client.data.batch_size)
        self.assertGreater(loss, 0.0)
        chex.assert_trees_all_close(
            ymir.utils.functions.ravel(self.client.params),
            ymir.utils.functions.ravel(self.params) - updates,
            atol=1e-8
        )
        weights, _, _ = self.client.step(self.params, return_weights=True)
        chex.assert_trees_all_close(
            ymir.utils.functions.ravel(self.client.params),
            weights,
            atol=1e-8
        )


class TestFedmax(unittest.TestCase):
    def setUp(self):
        dataset = Dataset()
        model = Net(dataset.classes)
        self.params = model.init(jax.random.PRNGKey(0), jnp.zeros(dataset.input_shape))
        opt = optax.sgd(0.1)
        loss = ymir.client.fedmax.loss(model)
        self.client = ymir.client.Client(self.params, opt, loss, dataset)

    def test_step(self):
        updates, loss, batch_size = self.client.step(self.params)
        self.assertEqual(batch_size, self.client.data.batch_size)
        self.assertGreater(loss, 0.0)
        chex.assert_trees_all_close(
            ymir.utils.functions.ravel(self.client.params),
            ymir.utils.functions.ravel(self.params) - updates,
            atol=1e-8
        )
        weights, _, _ = self.client.step(self.params, return_weights=True)
        chex.assert_trees_all_close(
            ymir.utils.functions.ravel(self.client.params),
            weights,
            atol=1e-8
        )



class TestFedprox(unittest.TestCase):
    def setUp(self):
        dataset = Dataset()
        self.params, _, loss = create_model(dataset.input_shape, dataset.classes)
        opt = ymir.client.fedprox.pgd(optax.sgd(0.1), 1)
        self.client = ymir.client.Client(self.params, opt, loss, dataset)

    def test_step(self):
        updates, loss, batch_size = self.client.step(self.params)
        self.assertEqual(batch_size, self.client.data.batch_size)
        self.assertGreater(loss, 0.0)
        chex.assert_trees_all_close(
            ymir.utils.functions.ravel(self.client.params),
            ymir.utils.functions.ravel(self.params) - updates,
            atol=1e-8
        )
        weights, _, _ = self.client.step(self.params, return_weights=True)
        chex.assert_trees_all_close(
            ymir.utils.functions.ravel(self.client.params),
            weights,
            atol=1e-8
        )

if __name__ == '__main__':
    unittest.main()
