import sys
from functools import partial
from itertools import product
import pickle
import pandas as pd
from absl import app

import jax
import jax.flatten_util
import jax.numpy as jnp
import optax
import haiku as hk

from tqdm import trange

import ymir

import metrics
import datasets


@jax.jit
def euclid_dist(a, b):
    return jnp.sqrt(jnp.sum((a - b)**2, axis=-1))

def unzero(x):
    return max(x, sys.float_info.epsilon)


def main(_):
    adv_percent = [0.1, 0.3, 0.5, 0.8]
    onoff_results = pd.DataFrame(columns=["algorithm", "attack", "dataset"] + [f"{p} mean asr" for p in adv_percent] + [f"{p} std asr" for p in adv_percent])
    VICTIM = 0
    print("Starting up...")
    for DATASET, IID in product(['mnist', 'kddcup99', 'cifar10'], [False, True]):
        DS = datasets.load(DATASET)
        for ALG in ["fed_avg", "krum", "std_dagmm", "viceroy"]:
            for ADV in ["labelflip", "onoff labelflip", "scaling backdoor", "onoff freerider", "bad mouther", "good mouther"]:
                if DATASET == 'kddcup99':
                    T = 20
                    ATTACK_FROM, ATTACK_TO = 0, 11
                else:
                    T = 10
                    ATTACK_FROM, ATTACK_TO = 0, 1
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

                    network = ymir.mp.network.Network()
                    network.add_controller("main", is_server=True)
                    for i in range(N):
                        network.add_host("main", ymir.scout.Collaborator(opt, opt_state, loss, data[i], 1))
                    for i in range(A):
                        c = ymir.scout.Collaborator(opt, opt_state, loss, data[i + N], batch_sizes[i + N])
                        if "labelflip" in ADV:
                            ymir.scout.adversaries.labelflipper.convert(c, DS, ATTACK_FROM, ATTACK_TO)
                        elif "backdoor" in ADV:
                            ymir.scout.adversaries.backdoor.convert(c, DS, DATASET, ATTACK_FROM, ATTACK_TO)
                        elif "freerider" in ADV:
                            ymir.scout.adversaries.freerider.convert(c, "delta", params)
                        if "onoff" in ADV:
                            ymir.scout.adversaries.onoff.convert(c)
                        network.add_host("main", c)
                    controller = network.get_controller("main")
                    if "scaling" in ADV:
                        controller.add_grad_transform(ymir.scout.adversaries.scaler.GradientTransform(network, params, ALG, A))
                    if "mouther" in ADV:
                        controller.add_grad_transform(ymir.scout.adversaries.mouther.GradientTransform(A, VICTIM, ADV))
                    if "onoff" not in ADV:
                        toggler = None
                    else:
                        toggler = ymir.scout.adversaries.onoff.GradientTransform(
                            network, params, ALG, controller.clients[-A:],
                            max_alpha=1/N if ALG in ['fed_avg', 'std_dagmm'] else 1,
                            sharp=ALG in ['fed_avg', 'std_dagmm', 'krum']
                        )
                        controller.add_grad_transform(toggler)

                    evaluator = metrics.measurer(net)

                    if "backdoor" in ADV:
                        test_eval = DS.get_iter(
                            "test",
                            map=partial({
                                "mnist": ymir.scout.adversaries.backdoor.mnist_backdoor_map,
                                "cifar10": ymir.scout.adversaries.backdoor.cifar10_backdoor_map,
                                "kddcup99": ymir.scout.adversaries.backdoor.kddcup99_backdoor_map
                            }[DATASET], ATTACK_FROM, ATTACK_FROM, no_label=True)
                        )

                    model = ymir.Coordinate(ALG, opt, opt_state, params, network)

                    results = metrics.create_recorder(['accuracy', 'asr'], train=True, test=True, add_evals=['attacking'])
                    results["asr"] = []

                    # Train/eval loop.
                    TOTAL_ROUNDS = 5_000
                    for round in (pbar := trange(TOTAL_ROUNDS)):
                        alpha, all_grads = model.step()
                        if round % 10 == 0:
                            attacking = toggler.attacking if toggler else True
                            metrics.record(results, evaluator, params, train_eval, test_eval, {'attacking': attacking}, attack_from=ATTACK_FROM, attack_to=ATTACK_TO)
                            if "freerider" in ADV:
                                if attacking:
                                    if ALG == "krum":
                                        results['asr'].append(alpha[-A:].mean())
                                    else:
                                        results['asr'].append(jnp.minimum(alpha[-A:].mean() / (1 / (alpha > 0).sum()), 1))
                                else:
                                    results['asr'].append(0.0)
                            elif "mouther" in ADV:
                                if (alpha[-A:] < 0.0001).all():
                                    asr = -1 if alpha[VICTIM] < 0.0001 else -2
                                else:
                                    theta = jax.flatten_util.ravel_pytree(ymir.garrison.sum_grads(all_grads))[0]
                                    vicdel = euclid_dist(jax.flatten_util.ravel_pytree(all_grads[VICTIM])[0], theta)
                                    if "good" in ADV:
                                        numerator = min(euclid_dist(jnp.array([jax.flatten_util.ravel_pytree(g)[0] for g in all_grads]), theta))
                                        asr = unzero(numerator) / unzero(vicdel)
                                    else:
                                        asr = unzero(vicdel) / unzero(max(euclid_dist(jnp.array([jax.flatten_util.ravel_pytree(g)[0] for g in all_grads]), theta)))
                                results['asr'].append(asr)
                            else:
                                results["asr"].append(results["test asr"][-1])
                            pbar.set_postfix({'ACC': f"{results['test accuracy'][-1]:.3f}", 'ASR': f"{results['asr'][-1]:.3f}", 'ATT': attacking})
                    results = metrics.finalize(results)
                    cur[f"{acal} mean asr"] = results['asr'].mean()
                    cur[f"{acal} std asr"] = results['asr'].std()
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


if __name__ == "__main__":
    app.run(main)