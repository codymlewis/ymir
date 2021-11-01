from functools import partial
import pickle
import pandas as pd

import jax
import jax.flatten_util
import optax
import haiku as hk

from tqdm import trange

import ymir

import metrics


if __name__ == "__main__":
    adv_percent = [0.1, 0.3, 0.5, 0.8]
    onoff_results = pd.DataFrame(columns=["algorithm", "attack", "dataset"] + [f"{p} mean asr" for p in adv_percent] + [f"{p} std asr" for p in adv_percent])
    print("Starting up...")
    IID = False
    for DATASET in ['mnist', 'kddcup99', 'cifar10']:
        DS = ymir.mp.datasets.load(DATASET)
        for ALG in ["foolsgold", "krum", "std_dagmm", "viceroy"]:
            for ADV in ["onoff labelflip", "labelflip", "scaling backdoor"]:
                if DATASET == 'kddcup99':
                    T = 20
                    ATTACK_FROM, ATTACK_TO = 0, 11
                else:
                    T = 10
                    ATTACK_FROM, ATTACK_TO = 0, 1
                ADV_CLASS = {
                    "labelflip": lambda o, _, b: ymir.scout.adversaries.LabelFlipper(o, DS, b, 1, ATTACK_FROM, ATTACK_TO),
                    "scaling backdoor": lambda o, _, b: ymir.scout.adversaries.Backdoor(o, DS, b, 1, ATTACK_FROM, ATTACK_TO),
                    "onoff labelflip": lambda o, d, b: ymir.scout.adversaries.OnOffLabelFlipper(o, d, DS, b, 1, ATTACK_FROM, ATTACK_TO),
                }[ADV]
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
                        con_class= {
                            "onoff": partial(
                                ymir.scout.adversaries.OnOffController,
                                alg=ALG,
                                num_adversaries=A,
                                max_alpha=1/N if ALG in ['fed_avg', 'std_dagmm'] else 1,
                                sharp=ALG in ['fed_avg', 'std_dagmm', 'krum']
                            ),
                            "labelflip": ymir.mp.network.Controller,
                            "scaling": partial(ymir.scout.adversaries.ScalingController, alg=ALG, num_adversaries=A)
                        }[ADV.split()[0]],
                        is_server=True
                    )
                    for i in range(N):
                        network.add_host("main", ymir.scout.Collaborator(opt_state, data[i], 1))
                    for i in range(A):
                        network.add_host("main", ADV_CLASS(opt_state, data[i + N], batch_sizes[i + N]))
                    controller = network.get_controller("main")
                    if ADV == "labelflip":
                        controller.attacking = True
                    if type(controller).__name__ != "Controller":
                        controller.init(params)

                    evaluator = metrics.measurer(net)

                    if "backdoor" in ADV:
                        test_eval = DS.get_iter(
                            "test",
                            map=partial({
                                "MNIST": ymir.scout.adversaries.mnist_backdoor_map,
                                "CIFAR10": ymir.scout.adversaries.cifar10_backdoor_map,
                                "KDDCup99": ymir.scout.adversaries.kddcup_backdoor_map
                            }[DATASET], ATTACK_FROM, ATTACK_FROM, no_label=True)
                        )

                    model = ymir.Coordinate(ALG, opt, opt_state, params, network)

                    results = metrics.create_recorder(['accuracy', 'asr'], train=True, test=True, add_evals=['attacking'])

                    # Train/eval loop.
                    TOTAL_ROUNDS = 5_001
                    pbar = trange(TOTAL_ROUNDS)
                    for round in pbar:
                        if round % 10 == 0:
                            metrics.record(results, evaluator, model.params, train_eval, test_eval, {'attacking': controller.attacking}, attack_from=ATTACK_FROM, attack_to=ATTACK_TO)
                            pbar.set_postfix({'ACC': f"{results['test accuracy'][-1]:.3f}", 'ASR': f"{results['test asr'][-1]:.3f}", 'ATT': controller.attacking})

                        model.step()
                    results = metrics.finalize(results)
                    cur[f"{acal} mean asr"] = results['test asr'].mean()
                    cur[f"{acal} std asr"] = results['test asr'].std()
                    print()
                    print("=" * 150)
                    print(f"Server type {ALG}, Dataset {DATASET}, {A / (A + N):.2%} {ADV} adversaries, final accuracy: {results['test accuracy'][-1]:.3%}")
                    print(metrics.tabulate(results, TOTAL_ROUNDS))
                    print("=" * 150)
                    print()
                onoff_results = onoff_results.append(cur, ignore_index=True)
    print(onoff_results.to_latex())
    fn = "results.pkl"
    with open(fn, 'wb') as f:
        pickle.dump(results, f)
    print(f"Written results to {fn}")
