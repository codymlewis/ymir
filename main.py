import pickle

import jax
import jax.flatten_util
import jax.numpy as jnp
import optax
import haiku as hk

from tqdm import trange, tqdm

import chief
from chief.aggregation import foolsgold
from chief.aggregation import std_dagmm
import endpoints
import utils



if __name__ == "__main__":
    net = hk.without_apply_rng(hk.transform(lambda x: utils.nets.LeNet()(x)))
    # neta = hk.without_apply_rng(hk.transform(lambda x: utils.nets.LeNet().act(x)))
    opt = optax.sgd(0.01)
    server_update = chief.update(opt)
    client_update = endpoints.update(opt, utils.losses.cross_entropy_loss(net))
    # client_update = endpoints.update(opt, utils.losses.fedmax_loss(net, neta))
    evaluator = utils.metrics.measurer(net)

    N = 8
    A = 2
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
            # grads = endpoints.compression.fedzip.encode(grads, compress=False)
            all_grads.append(grads)

        # Server side collection of gradients
        # all_grads = chief.compression.fedzip.decode(params, all_grads)
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