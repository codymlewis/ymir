from dataclasses import dataclass
from typing import Mapping
import pickle

import haiku as hk
import jax
import jax.flatten_util
import jax.numpy as jnp
import optax

from tqdm import trange, tqdm

import chief
from chief.aggregations import foolsgold
import endpoints
import utils


if __name__ == "__main__":
    net_fn, net_act = utils.nets.lenet(get_acts=True)
    net = hk.without_apply_rng(hk.transform(net_fn))
    neta = hk.without_apply_rng(hk.transform(net_act))
    opt = optax.sgd(0.01)
    server_update = chief.update(opt)
    client_update = endpoints.update(opt, utils.losses.fedmax_loss(net, neta))
    evaluator = utils.metrics.measurer(net)

    N = 7
    A = 3
    ATTACK_FROM = 8
    ATTACK_TO = 2

    train_eval = chief.load_dataset("train", batch_size=10000)
    test_eval = chief.load_dataset("test", batch_size=10000)

    params = net.init(jax.random.PRNGKey(42), next(test_eval))
    opt_state = opt.init(params)

    clients = [endpoints.Client(i, opt_state, 8) for i in range(N)]
    clients += [endpoints.adversaries.OnOff(i + N, opt_state, 8, ATTACK_FROM, ATTACK_TO) for i in range(A)]
    server = foolsgold.Server(jnp.zeros((len(clients), jax.flatten_util.ravel_pytree(params)[0].shape[0])), 1.0)

    attacking = False
    results = utils.metrics.create_recorder(['accuracy', 'asr'], train=True, test=True, add_evals=['attacking'])

    # Train/eval loop.
    for round in trange(10_001):
        if round % 10 == 0:
            utils.metrics.record(results, evaluator, params, train_eval, test_eval, {'attacking': attacking}, attack_from=ATTACK_FROM, attack_to=ATTACK_TO)
            if round % 100 == 0:
                tqdm.write(f"[Round {round}] " + ' '.join([f"{k}: {v[-1]}" for k, v in results.items()]))

        # Client side training
        all_grads = []
        for client in clients:
            grads, client.opt_state = client_update(params, client.opt_state, next(client.data))
            all_grads.append(grads)

        # Server side collection of gradients
        server.histories = foolsgold.update(server.histories, all_grads)
        alpha = foolsgold.scale(server.histories, server.kappa)
        all_grads = chief.apply_scale(alpha, all_grads)

        # Adversary interception and decision
        if endpoints.adversaries.should_toggle(clients, A, alpha, attacking):
            attacking = not attacking
            for c in clients[-A:]:
                c.toggle()

        # Server side aggregation
        params, opt_state = server_update(params, opt_state, chief.sum_grads(all_grads))
    fn = "results.pkl"
    with open(fn, 'wb') as f:
        pickle.dump(results, f)
    print(f"Written results to {fn}")