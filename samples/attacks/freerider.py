"""
Example of federated averaging on the MNIST dataset
"""

import haiku as hk
import hkzoo
import jax
import numpy as np
import optax
import tenjin
from tqdm import trange

import ymir

if __name__ == "__main__":
    print("Setting up the system...")
    num_honest = 10
    num_adversaries = 5
    num_clients = num_honest - num_adversaries
    rng = np.random.default_rng(0)

    # Setup the dataset
    dataset = ymir.mp.datasets.Dataset(*tenjin.load('mnist'))
    batch_sizes = [8 for _ in range(num_honest)]
    data = dataset.fed_split(batch_sizes, ymir.mp.distributions.lda, rng)
    train_eval = dataset.get_iter("train", 10_000, rng=rng)
    test_eval = dataset.get_iter("test", rng=rng)

    # Setup the network
    net = hk.without_apply_rng(hk.transform(lambda x: hkzoo.LeNet_300_100(dataset.classes, x)))
    client_opt = optax.sgd(0.1)
    params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])
    client_opt_state = client_opt.init(params)
    loss = ymir.mp.losses.cross_entropy_loss(net, dataset.classes)
    network = ymir.mp.network.Network()
    network.add_controller("main", server=True)
    for i in range(num_clients):
        network.add_host("main", ymir.regiment.Scout(client_opt, client_opt_state, loss, data[i], 1))
    for i in range(num_adversaries):
        c = ymir.regiment.Scout(client_opt, client_opt_state, loss, data[i + num_clients], 1)
        ymir.fritz.freerider.convert(c, "advanced delta", params)
        network.add_host("main", c)

    server_opt = optax.sgd(1)
    server_opt_state = server_opt.init(params)
    model = ymir.garrison.fedavg.Captain(params, server_opt, server_opt_state, network, rng)
    meter = ymir.mp.metrics.Neurometer(net, {'train': train_eval, 'test': test_eval})

    print("Done, beginning training.")

    # Train/eval loop.
    for r in (pbar := trange(5000)):
        if r % 10 == 0:
            results = meter.measure(model.params, ['test'])
            pbar.set_postfix({'ACC': f"{results['test acc']:.3f}"})
        model.step()