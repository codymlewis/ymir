"""
Example of parameter poisoning on Federated Averaging
"""

import haiku as hk
import hkzoo
import jax
import numpy as np
import optax
import tenjin
from tqdm import trange

import tfymir

if __name__ == "__main__":
    print("Setting up the system...")
    num_honest = 10
    num_adversaries = 2
    num_clients = num_honest - num_adversaries
    attack = "smp"
    rng = np.random.default_rng(0)

    # Setup the dataset
    dataset = tfymir.mp.datasets.Dataset(*tenjin.load('mnist'))
    batch_sizes = [8 for _ in range(num_honest)]
    data = dataset.fed_split(batch_sizes, tfymir.mp.distributions.lda, rng)
    train_eval = dataset.get_iter("train", 10_000, rng=rng)
    test_eval = dataset.get_iter("test", rng=rng)

    # Setup the network
    net = hk.without_apply_rng(hk.transform(lambda x: hkzoo.LeNet_300_100(dataset.classes, x)))
    client_opt = optax.sgd(0.01)
    params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])
    client_opt_state = client_opt.init(params)
    loss = tfymir.mp.losses.cross_entropy_loss(net, dataset.classes)
    network = tfymir.mp.network.Network()
    network.add_controller("main", server=True)
    for i in range(num_clients):
        network.add_host("main", tfymir.regiment.Scout(client_opt, client_opt_state, loss, data[i], 1))

    if attack == "smp":
        # Setup for the stealthy model poisoning attack
        adv_loss = tfymir.mp.losses.smp_loss(net, 10, loss, test_eval.X[:100], test_eval.y[:100], 10)
        adv_opt = tfymir.mp.optimizers.smp_opt(client_opt, 0.0001)
        adv_opt_state = adv_opt.init(params)
        for i in range(num_adversaries):
            c = tfymir.regiment.Scout(adv_opt, adv_opt_state, adv_loss, data[i + num_clients], 1)
            tfymir.fritz.labelflipper.convert(c, dataset, 0, 1)
            network.add_host("main", c)
    else:
        # setup for the alternating minimization attack
        stealth_data = dataset.get_iter("test", 8, rng=rng)
        for i in range(num_adversaries):
            c = tfymir.regiment.Scout(client_opt, client_opt_state, loss, data[i + num_clients], 10)
            tfymir.fritz.labelflipper.convert(c, dataset, 0, 1, 10)
            tfymir.fritz.alternating_minimization.convert(c, 1, 10, stealth_data)
            network.add_host("main", c)

    server_opt = optax.sgd(1)
    server_opt_state = server_opt.init(params)
    model = tfymir.garrison.fedavg.Captain(params, server_opt, server_opt_state, network, rng)
    meter = tfymir.mp.metrics.Neurometer(net, {'train': train_eval, 'test': test_eval})

    print("Done, beginning training.")

    # Train/eval loop.
    for r in (pbar := trange(5000)):
        if r % 10 == 0:
            results = meter.measure(model.params, ['test'], {'from': 0, 'to': 1, 'datasets': ['test']})
            pbar.set_postfix({'ACC': f"{results['test acc']:.3f}", 'ASR': f"{results['test asr']:.3f}"})
        model.step()
