import jax
import haiku as hk
import optax
from absl import app

from tqdm import trange

import ymir

"""
Example of federated averaging on the MNIST dataset
"""


def main(_):
    print("Setting up the system...")
    num_endpoints = 10

    # Setup the dataset
    dataset = ymir.mp.datasets.load('mnist')
    batch_sizes = [8 for _ in range(num_endpoints)]
    data = dataset.fed_split(batch_sizes, ymir.mp.distributions.lda)
    train_eval = dataset.get_iter("train", 10_000)
    test_eval = dataset.get_iter("test")

    # Setup the network
    net = hk.without_apply_rng(hk.transform(lambda x: ymir.mp.models.LeNet_300_100(dataset.classes)(x)))
    client_opt = optax.sgd(0.01)
    params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])
    client_opt_state = client_opt.init(params)
    loss = ymir.mp.losses.cross_entropy_loss(net, dataset.classes)
    network = ymir.mp.network.Network(client_opt, loss)
    network.add_controller("main", is_server=True)
    for d in data:
        network.add_host("main", ymir.scout.Collaborator(client_opt_state, d, 1))

    server_opt = optax.sgd(0.1)
    server_opt_state = server_opt.init(params)
    model = ymir.Coordinate("fed_avg", server_opt, server_opt_state, params, network)
    meter = ymir.mp.metrics.Neurometer(net, {'train': train_eval, 'test': test_eval})

    print("Done, beginning training.")

    # Train/eval loop.
    for r in (pbar := trange(5000)):
        if r % 10 == 0:
            results = meter.measure(model.params, ['test'])
            pbar.set_postfix({'ACC': f"{results['test']:.3f}"})
        model.step()


if __name__ == "__main__":
    app.run(main)