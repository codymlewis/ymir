from endpoints.adversaries import AdvController, OnOffController, ScalingController
from functools import partial
import pickle
import pandas as pd

import jax
import jax.flatten_util
import jax.numpy as jnp
import optax
import haiku as hk

from tqdm import trange

import chief
from chief.aggregation import foolsgold, std_dagmm
from chief.aggregation import fed_avg, krum, viceroy
import endpoints
import utils

def update_and_scale(alg, server, all_grads, round):
    if alg == 'fed_avg':
        alpha = fed_avg.scale(server.batch_sizes)
    elif alg == 'foolsgold':
        server.histories = foolsgold.update(server.histories, all_grads)
        alpha = foolsgold.scale(server.histories, server.kappa)
    elif alg == 'krum':
        alpha = krum.scale(all_grads, 3)
    elif alg == 'std_dagmm':
        server.params, server.opt_state = std_dagmm.update(server, all_grads)
        alpha = std_dagmm.scale(server.batch_sizes, all_grads, server)
    else:
        alpha = viceroy.scale(all_grads, server.histories, server.reps, round + 1)
        if round > 0:
            server.histories, server.reps = viceroy.update(server.histories, server.reps, all_grads)
        else:
            server.histories = viceroy.init(all_grads)
    return alpha, server

def update(alg, server, all_grads, round):
    if alg == 'foolsgold':
        server.histories = foolsgold.update(server.histories, all_grads)
    elif alg == 'std_dagmm':
        server.params, server.opt_state = std_dagmm.update(server, all_grads)
    elif alg == 'viceroy':
        if round > 0:
            server.histories, server.reps = viceroy.update(server.histories, server.reps, all_grads)
        else:
            server.histories = viceroy.init(all_grads)
    return server

def scale(alg, server, all_grads, round):
    if alg == 'fed_avg':
        alpha = fed_avg.scale(server.batch_sizes)
    elif alg == 'foolsgold':
        alpha = foolsgold.scale(server.histories, server.kappa)
    elif alg == 'krum':
        alpha = krum.scale(all_grads, 3)
    elif alg == 'std_dagmm':
        alpha = std_dagmm.scale(server.batch_sizes, all_grads, server)
    else:
        alpha = viceroy.scale(all_grads, server.histories, server.reps, round + 1)
    return alpha


if __name__ == "__main__":
    adv_percent = [0.1, 0.3, 0.5, 0.8]
    onoff_results = pd.DataFrame(columns=["algorithm", "attack", "dataset"] + [f"{p} mean asr" for p in adv_percent] + [f"{p} std asr" for p in adv_percent])
    print("Starting up...")
    IID = False
    for DATASET in [utils.datasets.KDDCup99, utils.datasets.CIFAR10]:
        DATASET = DATASET()
        for ALG in ["foolsgold", "krum"]:
            for ADV in ["onoff labelflip"]:
                if type(DATASET).__name__ == 'KDDCup99':
                    T = 20
                    ATTACK_FROM, ATTACK_TO = 0, 11
                else:
                    T = 10
                    ATTACK_FROM, ATTACK_TO = 0, 1
                ADV_CLASS = {
                    "labelflip": lambda o, _, b: endpoints.adversaries.LabelFlipper(o, DATASET, b, ATTACK_FROM, ATTACK_TO),
                    "scaling backdoor": lambda o, _, b: endpoints.adversaries.Backdoor(o, DATASET, b, ATTACK_FROM, ATTACK_TO),
                    "onoff labelflip": lambda o, d, b: endpoints.adversaries.OnOffLabelFlipper(o, d, DATASET, b, ATTACK_FROM, ATTACK_TO),
                }[ADV]
                SERVER_CLASS = {
                    'fed_avg': lambda: fed_avg.Server(jnp.array([c.batch_size for c in clients])),
                    'foolsgold': lambda: foolsgold.Server(len(clients), params, 1.0),
                    'krum': lambda: None,
                    'viceroy': lambda: viceroy.Server(len(clients), params),
                    'std_dagmm': lambda: std_dagmm.Server(jnp.array([c.batch_size for c in clients]), params)
                }[ALG]
                cur = {"algorithm": ALG, "attack": ADV, "dataset": type(DATASET).__name__}
                for acal in adv_percent:
                    print(f"Running {ALG} on {type(DATASET).__name__}{'-iid' if IID else ''} with {acal:.0%} {ADV} adversaries")
                    if type(DATASET).__name__ == 'CIFAR10':
                        net = hk.without_apply_rng(hk.transform(lambda x: utils.nets.ConvLeNet(DATASET.classes)(x)))
                    else:
                        net = hk.without_apply_rng(hk.transform(lambda x: utils.nets.LeNet(DATASET.classes)(x)))
                    opt = optax.sgd(0.01)
                    server_update = chief.update(opt)
                    client_update = endpoints.update(opt, utils.losses.cross_entropy_loss(net, DATASET.classes))
                    evaluator = utils.metrics.measurer(net)

                    A = int(T * acal)
                    N = T - A
                    batch_sizes = [8 for _ in range(N + A)]
                    data = DATASET.fed_split(batch_sizes, IID)

                    train_eval = DATASET.get_iter("train", 10_000)
                    test_eval = DATASET.get_iter("test")
                    if "backdoor" in ADV:
                        test_eval = DATASET.get_iter(
                            "test",
                            map=partial(endpoints.adversaries.backdoor_map, ATTACK_FROM, ATTACK_FROM)
                        )

                    params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])
                    opt_state = opt.init(params)

                    clients = [endpoints.Client(opt_state, data[i],  batch_sizes[i]) for i in range(N)]
                    clients += [ADV_CLASS(opt_state, data[i + N], batch_sizes[i + N]) for i in range(A)]
                    server = SERVER_CLASS()

                    results = utils.metrics.create_recorder(['accuracy', 'asr'], train=True, test=True, add_evals=['attacking'])

                    controller = {
                        "onoff": partial(OnOffController, max_alpha=1/N if ALG in ['fed_avg', 'std_dagmm'] else 1, sharp=ALG in ['fed_avg', 'std_dagmm', 'krum']),
                        "labelflip": AdvController,
                        "scaling": ScalingController
                    }[ADV.split()[0]](A, clients)

                    # Train/eval loop.
                    TOTAL_ROUNDS = 5_001
                    pbar = trange(TOTAL_ROUNDS)
                    for round in pbar:
                        if round % 10 == 0:
                            utils.metrics.record(results, evaluator, params, train_eval, test_eval, {'attacking': controller.attacking}, attack_from=ATTACK_FROM, attack_to=ATTACK_TO)
                            pbar.set_postfix({'ACC': f"{results['test accuracy'][-1]:.3f}", 'ASR': f"{results['test asr'][-1]:.3f}", 'ATT': controller.attacking})

                        # Client side training
                        all_grads = []
                        for client in clients:
                            grads, client.opt_state = client_update(params, client.opt_state, *next(client.data))
                            all_grads.append(grads)

                        # Server side collection of gradients
                        if ALG != "viceroy":
                            server = update(ALG, server, all_grads, round)
                        alpha = scale(ALG, server, all_grads, round)

                        # Adversary interception and decision
                        controller.intercept(alpha, all_grads)

                        if ADV.split()[0] == "scaling":
                            alpha = scale(ALG, server, all_grads, round)

                        if ALG == "viceroy":
                            server = update(ALG, server, all_grads, round)

                        all_grads = chief.apply_scale(alpha, all_grads)

                        # Server side aggregation
                        params, opt_state = server_update(params, opt_state, chief.sum_grads(all_grads))
                    results = utils.metrics.finalize(results)
                    cur[f"{acal} mean asr"] = results['test asr'].mean()
                    cur[f"{acal} std asr"] = results['test asr'].std()
                    print()
                    print("=" * 150)
                    print(f"Server type {ALG}, Dataset {type(DATASET).__name__}, {A / (A + N):.2%} {ADV} adversaries, final accuracy: {results['test accuracy'][-1]:.3%}")
                    print(utils.metrics.tabulate(results, TOTAL_ROUNDS))
                    print("=" * 150)
                    print()
                onoff_results = onoff_results.append(cur, ignore_index=True)
    print(onoff_results.to_latex())
        # with open('data.csv', 'a') as f:
        #     f.write(utils.metrics.csvline(type(DATASET).__name__, ALG,  A / (A + N), results, TOTAL_ROUNDS))
        # fn = "results.pkl"
        # with open(fn, 'wb') as f:
        #     pickle.dump(results, f)
        # print(f"Written results to {fn}")