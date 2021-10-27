import re
import jax
import haiku as hk
import optax

from tqdm import tqdm, trange

import ymir

"""
Example of FedProx on the MNIST dataset
"""


if __name__ == "__main__":
    # setup
    print("Setting up the system...")
    num_endpoints = 10
    dataset = ymir.mp.datasets.load('mnist')
    batch_sizes = [8 for _ in range(num_endpoints)]
    data = dataset.fed_split(batch_sizes, [[i % 10] for i in range(num_endpoints)])
    train_eval = dataset.get_iter("train", 10_000)
    test_eval = dataset.get_iter("test")

    net = hk.without_apply_rng(hk.transform(lambda x: ymir.mp.models.LeNet_300_100(dataset.classes)(x)))
    opt = ymir.mp.optimizers.pgd(0.01, 1)
    params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])
    opt_state = opt.init(params)
    loss = ymir.mp.losses.cross_entropy_loss(net, dataset.classes)
    network = ymir.mp.network.Network(opt, loss)
    network.add_controller("main", is_server=True)
    for d in data:
        network.add_host("main", ymir.scout.Client(opt_state, d, 1))

    server_opt = optax.sgd(0.01)
    server_opt_state = server_opt.init(params)

    model = ymir.Coordinate("fed_avg", server_opt, server_opt_state, params, network)
    meter = ymir.mp.metrics.Neurometer(net, {'train': train_eval, 'test': test_eval}, ['accuracy'])

    print("Done, beginning training.")

    # Train/eval loop.
    TOTAL_ROUNDS = 5_001
    pbar = trange(TOTAL_ROUNDS)
    for round in pbar:
        results = meter.add_record(model.params)
        pbar.set_postfix({'ACC': f"{results['test accuracy']:.3f}"})
        model.fit()