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
    data = dataset.fed_split(batch_sizes, [[i % 10] for i in range(num_endpoints)])
    train_eval = dataset.get_iter("train", 10_000)
    test_eval = dataset.get_iter("test")

    # Setup the network
    net = hk.without_apply_rng(hk.transform(lambda x: ymir.mp.models.LeNet_300_100(dataset.classes)(x)))
    opt = optax.sgd(0.01)
    params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])
    opt_state = opt.init(params)
    loss = ymir.mp.losses.cross_entropy_loss(net, dataset.classes)
    network = ymir.mp.network.Network(opt, loss)
    network.add_controller("main", is_server=True)
    for d in data:
        network.add_host("main", ymir.scout.Collaborator(opt_state, d, 1))

    model = ymir.Coordinate("fed_avg", opt, opt_state, params, network)
    meter = ymir.mp.metrics.Neurometer(net, {'train': train_eval, 'test': test_eval}, ['accuracy'])

    print("Done, beginning training.")

    # Train/eval loop.
    for _ in (pbar := trange(5001)):
        results = meter.add_record(model.params)
        pbar.set_postfix({'ACC': f"{results['test accuracy']:.3f}"})
        model.step()


if __name__ == "__main__":
    app.run(main)