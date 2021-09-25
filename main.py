import pickle

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


if __name__ == "__main__":
    ALG = "foolsgold"
    DATASET = utils.datasets.KDDCup99()
    IID = False
    if type(DATASET).__name__ == 'KDDCup99':
        T = 20
        ATTACK_FROM, ATTACK_TO = 0, 11
    else:
        T = 10
        ATTACK_FROM, ATTACK_TO = 0, 1
    ADV = "label flip"
    ADV_CLASS = {
        "label flip": lambda o, _, b: endpoints.adversaries.LabelFlipper(o, DATASET, b, ATTACK_FROM, ATTACK_TO),
        "backdoor": lambda o, _, b: endpoints.adversaries.Backdoor(o, DATASET, b, ATTACK_FROM, ATTACK_TO),
        "on off label flip": lambda o, d, b: endpoints.adversaries.OnOffLabelFlipper(o, d, DATASET, b, ATTACK_FROM, ATTACK_TO),
        "on off free rider": endpoints.adversaries.OnOffFreeRider
    }[ADV]
    SERVER_CLASS = {
        'fed_avg': lambda: fed_avg.Server(jnp.array([c.batch_size for c in clients])),
        'foolsgold': lambda: foolsgold.Server(len(clients), params, 1.0),
        'krum': lambda: None,
        'viceroy': lambda: viceroy.Server(len(clients), params),
        'std_dagmm': lambda: std_dagmm.Server(jnp.array([c.batch_size for c in clients]), params)
    }[ALG]
    for acal in [0.1, 0.3, 0.5, 0.8]:
        print(f"Running {ALG} on {type(DATASET).__name__}{'-iid' if IID else ''} with {acal:.0%} {ADV} adversaries")
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

        params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])
        opt_state = opt.init(params)

        clients = [endpoints.Client(opt_state, data[i],  batch_sizes[i]) for i in range(N)]
        clients += [ADV_CLASS(opt_state, data[i + N], batch_sizes[i + N]) for i in range(A)]
        server = SERVER_CLASS()

        attacking = "on off" not in ADV
        results = utils.metrics.create_recorder(['accuracy', 'asr'], train=True, test=True, add_evals=['attacking'])

        MAX_ALPHA = 1/N if ALG in ['fed_avg', 'std_dagmm'] else 1
        SHARP = ALG in ['fed_avg', 'std_dagmm', 'krum']

        # Train/eval loop.
        TOTAL_ROUNDS = 5_1
        pbar = trange(TOTAL_ROUNDS)
        for round in pbar:
            if round % 10 == 0:
                utils.metrics.record(results, evaluator, params, train_eval, test_eval, {'attacking': attacking}, attack_from=ATTACK_FROM, attack_to=ATTACK_TO)
                pbar.set_postfix({'ACC': f"{results['test accuracy'][-1]:.3f}", 'ASR': f"{results['test asr'][-1]:.3f}", 'ATT': attacking})

            # Client side training
            all_grads = []
            for client in clients:
                grads, client.opt_state = client_update(params, client.opt_state, *next(client.data))
                all_grads.append(grads)

            # Server side collection of gradients
            alpha, server = update_and_scale(ALG, server, all_grads, round)
            all_grads = chief.apply_scale(alpha, all_grads)

            # Adversary interception and decision
            if "on off" in ADV:
                if endpoints.adversaries.should_toggle(A, alpha, attacking, MAX_ALPHA, sharp=SHARP):
                    attacking = not attacking
                    for c in clients[-A:]:
                        c.toggle()

            # Server side aggregation
            params, opt_state = server_update(params, opt_state, chief.sum_grads(all_grads))
        results = utils.metrics.finalize(results)
        print()
        print("=" * 150)
        print(f"Server type {ALG}, Dataset {type(DATASET).__name__}, {A / (A + N):.2%} {ADV} adversaries, final accuracy: {results['test accuracy'][-1]:.3%}")
        print(utils.metrics.tabulate(results, TOTAL_ROUNDS))
        print("=" * 150)
        print()
        # with open('data.csv', 'a') as f:
        #     f.write(utils.metrics.csvline(type(DATASET).__name__, ALG,  A / (A + N), results, TOTAL_ROUNDS))
        # fn = "results.pkl"
        # with open(fn, 'wb') as f:
        #     pickle.dump(results, f)
        # print(f"Written results to {fn}")