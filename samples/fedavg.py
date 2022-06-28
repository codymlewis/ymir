import os

import datasets
import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm

import ymir


class LeNet(nn.Module):

    @nn.compact
    def __call__(self, x):
        return nn.Sequential(
            [
                lambda x: einops.rearrange(x, "b w h c -> b (w h c)"),
                nn.Dense(300), nn.relu,
                nn.Dense(100), nn.relu,
                nn.Dense(10), nn.softmax
            ]
        )(x)


def loss(model):

    @jax.jit
    def _loss(params, X, y):
        logits = jnp.clip(model.apply(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))

    return _loss


def accuracy(model, params, X, y):
    return jnp.mean(jnp.argmax(model.apply(params, X), axis=-1) == y)


def load_dataset():
    ds = datasets.load_dataset('mnist')
    ds = ds.map(
        lambda e: {
            'X': einops.rearrange(np.array(e['image'], dtype=np.float32) / 255, "h (w c) -> h w c", c=1),
            'Y': e['label']
        },
        remove_columns=['image', 'label']
    )
    features = ds['train'].features
    features['X'] = datasets.Array3D(shape=(28, 28, 1), dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['test'] = ds['test'].cast(features)
    ds.set_format('numpy')
    return ds


if __name__ == "__main__":
    num_clients = 10
    dataset = ymir.utils.datasets.Dataset(load_dataset())
    batch_sizes = [32 for _ in range(num_clients)]
    data = dataset.fed_split(batch_sizes, ymir.utils.distributions.lda, in_memory=True)
    test_eval = dataset.get_iter("test", 10_000)

    model = LeNet()
    params = model.init(jax.random.PRNGKey(42), np.zeros((32,) + dataset.input_shape))

    network = ymir.utils.network.Network()
    for d in data:
        network.add_client(ymir.client.Client(params, optax.sgd(0.1), loss(model.clone()), d))
    server = ymir.server.fedavg.Server(network, params)
    for r in (p := tqdm.trange(3750)):
        loss_val = server.step()
        p.set_postfix_str(f"loss: {loss_val:.3f}")
    print(
        f"Test loss: {loss(model)(server.params, *next(test_eval)):.3f}, Test accuracy: {accuracy(model, server.params, *next(test_eval)):.3%}"
    )
