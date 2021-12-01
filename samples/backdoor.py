from functools import partial

import numpy as np
import jax
import haiku as hk
import optax
from absl import app

from tqdm import trange

import ymir

"""
Example of parameter poisoning on Federated Averaging
"""


def main(_):
    print("Setting up the system...")
    num_endpoints = 10
    num_adversaries = 2
    num_clients = num_endpoints - num_adversaries
    rng = np.random.default_rng(0)

    # Setup the dataset
    dataset = ymir.mp.datasets.load('mnist')
    batch_sizes = [8 for _ in range(num_endpoints)]
    data = dataset.fed_split(batch_sizes, partial(ymir.mp.distributions.lda, alpha=0.05), rng)
    train_eval = dataset.get_iter("train", 10_000, rng=rng)
    test_eval = dataset.get_iter("test", rng=rng)

    # Setup the network
    net = hk.without_apply_rng(hk.transform(lambda x: ymir.mp.models.LeNet_300_100(dataset.classes)(x)))
    client_opt = optax.sgd(0.01)
    params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])
    client_opt_state = client_opt.init(params)
    loss = ymir.mp.losses.cross_entropy_loss(net, dataset.classes)
    network = ymir.mp.network.Network()
    network.add_controller("main", server=True)
    for i in range(num_clients):
        network.add_host("main", ymir.scout.Collaborator(client_opt, client_opt_state, loss, data[i], 1))
    
    # Setup for the constrain and scale attack
    adv_loss = ymir.mp.losses.constrain_cosine_loss(0.4, loss, client_opt, client_opt_state)
    for i in range(num_adversaries):
        c = ymir.scout.Collaborator(client_opt, client_opt_state, adv_loss, data[i + num_clients], 1)
        ymir.scout.adversaries.backdoor.convert(c, dataset, "mnist", 0, 1)
        network.add_host("main", c)
    network.get_controller("main").add_update_transform(ymir.scout.adversaries.scaler.GradientTransform(network, params, "foolsgold", num_adversaries))

    backdoor_eval = dataset.get_iter("test", map=partial(ymir.scout.adversaries.backdoor.mnist_backdoor_map, 0, 1, no_label=True))

    server_opt = optax.sgd(0.1)
    server_opt_state = server_opt.init(params)
    model = ymir.Coordinate("foolsgold", server_opt, server_opt_state, params, network, rng)
    meter = ymir.mp.metrics.Neurometer(net, {'train': train_eval, 'test': test_eval, 'backdoor': backdoor_eval})

    print("Done, beginning training.")

    # Train/eval loop.
    for r in (pbar := trange(5000)):
        if r % 10 == 0:
            results = meter.measure(model.params, ['test'], {'from': 0, 'to': 1, 'datasets': ['backdoor']})
            pbar.set_postfix({'ACC': f"{results['test acc']:.3f}", 'ASR': f"{results['backdoor asr']:.3f}"})
        model.step()


if __name__ == "__main__":
    app.run(main)