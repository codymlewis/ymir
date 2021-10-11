import jax
import haiku as hk
import optax

from tqdm import tqdm, trange

import ymir

"""
Example of federated averaging on the MNIST dataset
"""


if __name__ == "__main__":
    # setup
    print("Setting up the system...")
    num_endpoints = 10
    dataset = ymir.mp.datasets.MNIST()
    batch_sizes = [8 for _ in range(num_endpoints)]
    data = dataset.fed_split(batch_sizes, False)
    train_eval = dataset.get_iter("train", 10_000)
    test_eval = dataset.get_iter("test")

    net = hk.without_apply_rng(hk.transform(lambda x: ymir.mp.models.LeNet(dataset.classes)(x)))
    opt = optax.sgd(0.01)
    params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])
    opt_state = opt.init(params)
    loss = ymir.mp.losses.cross_entropy_loss(net, dataset.classes)
    network = ymir.mp.network.Network(opt, loss)
    network.add_controller("main", is_server=True)
    for d in data:
        network.add_host("main", ymir.scout.Client(opt_state, d))

    model = ymir.Coordinate("fed_avg", opt, opt_state, params, network)

    # metrics setup
    evaluator = ymir.mp.metrics.measurer(net)
    results = ymir.mp.metrics.create_recorder(['accuracy'], train=True, test=True)

    print("Done, beginning training.")

    # Train/eval loop.
    TOTAL_ROUNDS = 5_001
    pbar = trange(TOTAL_ROUNDS)
    for round in pbar:
        if round % 10 == 0:
            ymir.mp.metrics.record(results, evaluator, model.params, train_eval, test_eval)
            pbar.set_postfix({'ACC': f"{results['test accuracy'][-1]:.3f}"})
        model.fit()