from functools import partial

import jax
import haiku as hk
import optax

from tqdm import tqdm, trange

import ymir

"""
Evaluation of heterogeneous techniques applied to viceroy.
"""


if __name__ == "__main__":
    for test_alg in ["fedprox", "fedmax"]:
        num_endpoints = 20
        num_adversaries = round(num_endpoints * 0.3)
        num_clients = num_endpoints - num_adversaries
        attack_from, attack_to = 0, 11
        print(f"Setting up the {test_alg} system with {num_adversaries} adversaries on-off flipping {attack_from}->{attack_to}...")

        dataset = ymir.mp.datasets.load('kddcup99')
        batch_sizes = [8 for _ in range(num_endpoints)]
        data = dataset.fed_split(batch_sizes, [[(i + 1 if i >= 11 else i) % dataset.classes, 11] for i in range(num_endpoints)])
        train_eval = dataset.get_iter("train", 10_000)
        test_eval = dataset.get_iter("test")

        selected_model = lambda: ymir.mp.models.LeNet_300_100(dataset.classes)
        net = hk.without_apply_rng(hk.transform(lambda x: selected_model()(x)))
        net_act = hk.without_apply_rng(hk.transform(lambda x: selected_model().act(x)))
        opt = optax.sgd(0.01)
        params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])
        opt_state = opt.init(params)
        if test_alg == "fedmax":
            loss = ymir.mp.losses.fedmax_loss(net, net_act, dataset.classes)
            client_opt, client_opt_state = opt, opt_state
        else:
            loss = ymir.mp.losses.cross_entropy_loss(net, dataset.classes)
            client_opt = ymir.mp.optimizers.pgd(0.01, 1)
            client_opt_state = client_opt.init(params)
        
        network = ymir.mp.network.Network(client_opt, loss)
        network.add_controller(
            "main",
            con_class=partial(ymir.scout.adversaries.OnOffController, alg="viceroy", num_adversaries=num_adversaries, max_alpha=1, sharp=False),
            is_server=True
        )
        for i in range(num_clients):
            network.add_host("main", ymir.scout.Client(client_opt_state, data[i], 1))
        for i in range(num_adversaries):
            network.add_host(
                "main",
                ymir.scout.adversaries.OnOffLabelFlipper(client_opt_state, data[num_clients + i], dataset, batch_sizes[num_clients + i], 1, attack_from, attack_to)
            )
        (controller := network.get_controller("main")).init(params)

        model = ymir.Coordinate("viceroy", opt, opt_state, params, network)
        meter = ymir.mp.metrics.Neurometer(net, {'train': train_eval, 'test': test_eval}, ['accuracy', 'asr'], attack_from=attack_from, attack_to=attack_to)

        print("Done, beginning training.")

        # Train/eval loop.
        TOTAL_ROUNDS = 5_001
        pbar = trange(TOTAL_ROUNDS)
        for round in pbar:
            results = meter.add_record(model.params)
            pbar.set_postfix({'ACC': f"{results['test accuracy']:.3f}", 'ASR': f"{results['test asr']:.3f}", 'ATT': controller.attacking})
            model.fit()
        print(", ".join([f"mean {k}: {v.mean()}" for k, v in meter.get_results().items()]))