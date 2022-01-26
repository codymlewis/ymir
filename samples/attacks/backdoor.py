"""
Example of a backdoor attack on Federated Averaging
"""

from functools import partial

import numpy as np
import jax
import haiku as hk
import optax

from tqdm import trange

import hkzoo
import tenjin
import ymir

if __name__ == "__main__":
    print("Setting up the system...")
    num_endpoints = 10
    num_adversaries = 2
    num_clients = num_endpoints - num_adversaries
    rng = np.random.default_rng(0)

    # Setup the dataset
    dataset = ymir.mp.datasets.Dataset(*tenjin.load('mnist'))
    batch_sizes = [8 for _ in range(num_endpoints)]
    data = dataset.fed_split(batch_sizes, partial(ymir.mp.distributions.lda, alpha=0.05), rng)
    train_eval = dataset.get_iter("train", 10_000, rng=rng)
    test_eval = dataset.get_iter("test", rng=rng)

    # Setup the network
    net = hk.without_apply_rng(hk.transform(lambda x: hkzoo.LeNet_300_100(dataset.classes, x)))
    client_opt = optax.sgd(0.01)
    params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])
    client_opt_state = client_opt.init(params)
    loss = ymir.mp.losses.cross_entropy_loss(net, dataset.classes)
    network = ymir.mp.network.Network()
    network.add_controller("main", server=True)
    for i in range(num_clients):
        network.add_host("main", ymir.regiment.Scout(client_opt, client_opt_state, loss, data[i], 1))

    # Setup for the constrain and scale attack
    adv_loss = ymir.mp.losses.constrain_cosine_loss(0.4, loss, client_opt, client_opt_state)
    for i in range(num_adversaries):
        c = ymir.regiment.Scout(client_opt, client_opt_state, adv_loss, data[i + num_clients], 1)
        ymir.fritz.backdoor.convert(c, dataset, 0, 1, np.ones((5, 5, 1)))
        ymir.fritz.scaler.convert(c, num_endpoints)
        network.add_host("main", c)

    backdoor_eval = dataset.get_iter(
        "test", map=partial(ymir.fritz.backdoor.backdoor_map, 0, 1, np.ones((5, 5, 1)), no_label=True)
    )

    server_opt = optax.sgd(1)
    server_opt_state = server_opt.init(params)
    model = ymir.garrison.foolsgold.Captain(params, server_opt, server_opt_state, network, rng)
    meter = ymir.mp.metrics.Neurometer(net, {'train': train_eval, 'test': test_eval, 'backdoor': backdoor_eval})

    print("Done, beginning training.")

    # Train/eval loop.
    for r in (pbar := trange(5000)):
        if r % 10 == 0:
            results = meter.measure(model.params, ['test'], {'from': 0, 'to': 1, 'datasets': ['backdoor']})
            pbar.set_postfix({'ACC': f"{results['test acc']:.3f}", 'ASR': f"{results['backdoor asr']:.3f}"})
        model.step()
