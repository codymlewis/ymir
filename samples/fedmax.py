import re
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
    # setup
    print("Setting up the system...")
    num_endpoints = 20
    dataset = ymir.mp.datasets.load('kddcup99')
    batch_sizes = [64 for _ in range(num_endpoints)]
    data = dataset.fed_split(batch_sizes, ymir.mp.distributions.lda)
    train_eval = dataset.get_iter("train", 10_000)
    test_eval = dataset.get_iter("test")

    selected_model = lambda: ymir.mp.models.LeNet_300_100(dataset.classes)
    net = hk.without_apply_rng(hk.transform(lambda x: selected_model()(x)))
    net_act = hk.without_apply_rng(hk.transform(lambda x: selected_model().act(x)))
    opt = optax.sgd(0.01)
    params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])
    opt_state = opt.init(params)
    loss = ymir.mp.losses.fedmax_loss(net, net_act, dataset.classes)
    network = ymir.mp.network.Network()
    network.add_controller("main", server=True)
    for d in data:
        network.add_host("main", ymir.regiment.Scout(opt, opt_state, loss, d, 10))

    model = ymir.garrison.fedavg.Captain(params, opt, opt_state, network)
    meter = ymir.mp.metrics.Neurometer(net, {'train': train_eval, 'test': test_eval})

    print("Done, beginning training.")

    # Train/eval loop.
    for _ in (pbar := trange(500)):
        results = meter.measure(model.params, ['test'])
        pbar.set_postfix({'ACC': f"{results['test acc']:.3f}"})
        model.step()


if __name__ == "__main__":
    app.run(main)