"""
Some experiments evaluating the FLGuard algorithm from https://arxiv.org/abs/2101.02281
"""

import pickle

import numpy as np
import jax
import haiku as hk
import optax
from absl import app

from tqdm import trange

import ymir


def main(_):
    print("Setting up the system...")
    num_endpoints = 10
    rng = np.random.default_rng(0)

    # Setup the dataset
    dataset = ymir.mp.datasets.load('mnist')
    batch_sizes = [8 for _ in range(num_endpoints)]
    data = dataset.fed_split(batch_sizes, ymir.mp.distributions.lda, rng)
    train_eval = dataset.get_iter("train", 10_000, rng=rng)
    test_eval = dataset.get_iter("test", rng=rng)

    # Setup the network
    net = hk.without_apply_rng(hk.transform(lambda x: ymir.mp.models.LeNet_300_100(dataset.classes, x)))
    client_opt = optax.sgd(0.01)
    params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])
    client_opt_state = client_opt.init(params)
    loss = ymir.mp.losses.cross_entropy_loss(net, dataset.classes)
    network = ymir.mp.network.Network()
    network.add_controller("main", server=True)
    # endpoint_epochs = np.round(rng.uniform(20, 30, size=num_endpoints)).astype(int)
    endpoint_epochs = [1 for _ in range(num_endpoints)]
    print(f"Endpoint epochs: {endpoint_epochs}")
    for e, d in zip(endpoint_epochs, data):
        network.add_host("main", ymir.regiment.Scout(client_opt, client_opt_state, loss, d, e))

    server_opt = optax.sgd(0.1)
    server_opt_state = server_opt.init(params)
    model = ymir.garrison.flguard.Captain(params, server_opt, server_opt_state, network, rng)
    meter = ymir.mp.metrics.Neurometer(net, {'train': train_eval, 'test': test_eval})

    print("Done, beginning training.")

    # Train/eval loop.
    for _ in (pbar := trange(round(5000 / np.mean(endpoint_epochs)))):  # equalize steps of training
        results = meter.measure(model.params, ['test'])
        pbar.set_postfix({'ACC': f"{results['test acc']:.3f}"})
        model.step()

    with open('results.pkl', 'wb') as f:
        pickle.dump(meter.conclude(), f)
    print("Results saved to results.pkl")


if __name__ == "__main__":
    app.run(main)