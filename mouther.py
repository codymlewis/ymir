from endpoints.adversaries import AdvController, MoutherController, OnOffController, OnOffFRController, ScalingController, FRController
from functools import partial
import pickle
import sys

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


@jax.jit
def euclid_dist(a, b):
    return jnp.sqrt(jnp.sum((a - b)**2, axis=-1))

def unzero(x):
    return max(x, sys.float_info.epsilon)


if __name__ == "__main__":
    print("Starting up...")
    ALG = "foolsgold"
    DATASET = utils.datasets.KDDCup99()
    IID = False
    if type(DATASET).__name__ == 'KDDCup99':
        T = 20
    else:
        T = 10
    VICTIM = 0
    ADV = "good mouther"
    ADV_CLASS = endpoints.Client
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

        results = utils.metrics.create_recorder(['accuracy'], train=True, test=True, add_evals=['attacking'])
        results['asr'] = []

        controller = MoutherController(A, VICTIM, ADV)

        # Train/eval loop.
        TOTAL_ROUNDS = 5_001
        pbar = trange(TOTAL_ROUNDS)
        for round in pbar:
            # Client side training
            all_grads = []
            for client in clients:
                grads, client.opt_state = client_update(params, client.opt_state, *next(client.data))
                all_grads.append(grads)

            # Adversary interception and decision
            controller.intercept(all_grads)


            # Server side collection of gradients
            if ALG != "viceroy":
                server = update(ALG, server, all_grads, round)
            alpha = scale(ALG, server, all_grads, round)


            if ALG == "viceroy":
                server = update(ALG, server, all_grads, round)

            all_grads = chief.apply_scale(alpha, all_grads)

            if round % 10 == 0:
                theta = jax.flatten_util.ravel_pytree(chief.sum_grads(all_grads))[0]
                vicdel = euclid_dist(jax.flatten_util.ravel_pytree(all_grads[VICTIM])[0], theta)
                if "good" in ADV:
                    numerator = min(euclid_dist(jnp.array([jax.flatten_util.ravel_pytree(g)[0] for g in all_grads]), theta))
                    asr = unzero(numerator) / unzero(vicdel)
                else:
                    asr = unzero(vicdel) / unzero(max(euclid_dist(jnp.array([jax.flatten_util.ravel_pytree(g)[0] for g in all_grads]), theta)))
                results['asr'].append(asr)
                utils.metrics.record(results, evaluator, params, train_eval, test_eval, {'attacking': controller.attacking})
                pbar.set_postfix({'ACC': f"{results['test accuracy'][-1]:.3f}", 'ASR': f"{results['asr'][-1]:.3f}", 'ATT': controller.attacking})

            # Server side aggregation
            params, opt_state = server_update(params, opt_state, chief.sum_grads(all_grads))
        results = utils.metrics.finalize(results)
        print()
        print("=" * 150)
        print(f"Server type {ALG}, Dataset {type(DATASET).__name__}, {A / (A + N):.2%} {ADV} adversaries, final accuracy: {results['test accuracy'][-1]:.3%}")
        print(utils.metrics.tabulate(results, TOTAL_ROUNDS))
        print("=" * 150)
        print()