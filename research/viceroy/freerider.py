from functools import partial
import pickle
import pandas as pd
from absl import app

import jax
import jax.numpy as jnp
import jax.flatten_util
import optax
import haiku as hk

from tqdm import trange

import ymir

import metrics
import datasets


def main(_):
    adv_percent = [0.1, 0.3, 0.5, 0.8]
    onoff_results = pd.DataFrame(columns=["algorithm", "attack", "dataset"] + [f"{p} mean asr" for p in adv_percent] + [f"{p} std asr" for p in adv_percent])
    print("Starting up...")
    IID = False
    for DATASET in ['mnist', 'kddcup99', 'cifar10']:
        DS = datasets.load(DATASET)
        for ALG in ["foolsgold", "krum", "std_dagmm", "viceroy"]:
            for ADV in ["onoff freerider"]:
                if DATASET == 'kddcup99':
                    T = 20
                else:
                    T = 10
                cur = {"algorithm": ALG, "attack": ADV, "dataset": DATASET}
                for acal in adv_percent:
                    print(f"Running {ALG} on {DATASET}{'-iid' if IID else ''} with {acal:.0%} {ADV} adversaries")
                    if DATASET == 'cifar10':
                        net = hk.without_apply_rng(hk.transform(lambda x: ymir.mp.models.ConvLeNet(DS.classes)(x)))
                    else:
                        net = hk.without_apply_rng(hk.transform(lambda x: ymir.mp.models.LeNet_300_100(DS.classes)(x)))

                    train_eval = DS.get_iter("train", 10_000)
                    test_eval = DS.get_iter("test")
                    opt = optax.sgd(0.01)
                    params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])
                    opt_state = opt.init(params)
                    loss = ymir.mp.losses.cross_entropy_loss(net, DS.classes)

                    A = int(T * acal)
                    N = T - A
                    batch_sizes = [8 for _ in range(N + A)]
                    if IID:
                        data = DS.fed_split(batch_sizes)
                    else:
                        if DATASET != 'kddcup99':
                            data = DS.fed_split(batch_sizes, [[i % 10] for i in range(T)])
                        else:
                            data = DS.fed_split(batch_sizes, [[(i + 1 if i >= 11 else i) % DS.classes, 11] for i in range(T)])

                    network = ymir.mp.network.Network(opt, loss)
                    network.add_controller(
                        "main",
                        con_class = partial({
                            "onoff freerider": partial(
                                ymir.scout.adversaries.OnOffFRController,
                                alg=ALG,
                                num_adversaries=A,
                                max_alpha=1/N if ALG in ['fed_avg', 'std_dagmm'] else 1,
                                sharp=ALG in ['fed_avg', 'std_dagmm', 'krum']
                            ),
                            "freerider": partial(ymir.scout.adversaries.FRController, attack_type="delta"),
                        }[ADV], num_adversaries=A, params=params),
                        is_server=True
                    )

                    for i in range(T):
                        network.add_host("main", ymir.scout.Collaborator(opt_state, data[i], 1))
                    controller = network.get_controller("main")
                    if "onoff" in ADV:
                        controller.init(params)

                    evaluator = metrics.measurer(net)

                    model = ymir.Coordinate(ALG, opt, opt_state, params, network)

                    results = metrics.create_recorder(['accuracy'], train=True, test=True, add_evals=['attacking'])
                    results['asr'] = []

                    # Train/eval loop.
                    TOTAL_ROUNDS = 5_001
                    pbar = trange(TOTAL_ROUNDS)
                    for round in pbar:
                        alpha, _ = model.step()

                        if round % 10 == 0:
                            if controller.attacking:
                                if ALG == "krum":
                                    results['asr'].append(alpha[-A:].mean())
                                else:
                                    results['asr'].append(jnp.minimum(alpha[-A:].mean() / (1 / (alpha > 0).sum()), 1))
                            else:
                                results['asr'].append(0.0)
                            metrics.record(results, evaluator, params, train_eval, test_eval, {'attacking': controller.attacking})
                            pbar.set_postfix({'ACC': f"{results['test accuracy'][-1]:.3f}", 'ASR': f"{results['asr'][-1]:.3f}", 'ATT': controller.attacking})

                    results = metrics.finalize(results)
                    cur[f"{acal} mean asr"] = results['test asr'].mean()
                    cur[f"{acal} std asr"] = results['test asr'].std()
                    print()
                    print("=" * 150)
                    print(f"Server type {ALG}, Dataset {type(DATASET).__name__}, {A / (A + N):.2%} {ADV} adversaries, final accuracy: {results['test accuracy'][-1]:.3%}")
                    print(metrics.tabulate(results, TOTAL_ROUNDS))
                    print("=" * 150)
                    print()
                onoff_results = onoff_results.append(cur, ignore_index=True)
    print(onoff_results.to_latex())
    fn = "results.pkl"
    with open(fn, 'wb') as f:
        pickle.dump(results, f)
    print(f"Written results to {fn}")


if __name__ == "__main__":
    app.run(main)