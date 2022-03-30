"""
Example of federated averaging on the MNIST dataset
"""

import haiku as hk
import hkzoo
import jax
import optax
import tenjin
from tqdm import trange

import tfymir

if __name__ == "__main__":
    # setup
    print("Setting up the system...")
    num_clients = 20
    dataset = tfymir.mp.datasets.Dataset(*tenjin.load('kddcup99'))
    batch_sizes = [64 for _ in range(num_clients)]
    data = dataset.fed_split(batch_sizes, tfymir.mp.distributions.lda)
    train_eval = dataset.get_iter("train", 10_000)
    test_eval = dataset.get_iter("test")

    selected_model = lambda x, a: hkzoo.LeNet_300_100(dataset.classes, x, a)
    net = hk.without_apply_rng(hk.transform(lambda x: selected_model(x, False)))
    net_act = hk.without_apply_rng(hk.transform(lambda x: selected_model(x, True)))
    opt = optax.sgd(0.01)
    params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])
    opt_state = opt.init(params)
    loss = tfymir.mp.losses.fedmax_loss(net, net_act, dataset.classes)
    network = tfymir.mp.network.Network()
    network.add_controller("main", server=True)
    for d in data:
        network.add_host("main", tfymir.regiment.Scout(opt, opt_state, loss, d, 10))

    server_opt = optax.sgd(1)
    server_opt_state = server_opt.init(params)
    model = tfymir.garrison.fedavg.Captain(params, server_opt, server_opt_state, network)
    meter = tfymir.mp.metrics.Neurometer(net, {'train': train_eval, 'test': test_eval})

    print("Done, beginning training.")

    # Train/eval loop.
    for _ in (pbar := trange(500)):
        results = meter.measure(model.params, ['test'])
        pbar.set_postfix({'ACC': f"{results['test acc']:.3f}"})
        model.step()
