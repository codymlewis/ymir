from functools import partial

import pandas as pd

import jax
import jax.numpy as jnp
import haiku as hk
import optax
from absl import app

from tqdm import tqdm, trange

import ymir

import utils

"""
Evaluation of heterogeneous techniques applied to viceroy.
"""


def main(_):
    adv_percent = [0.3, 0.5]
    for comp_alg in ["fedmax", "fedprox", "fedzip", "ae"]:
        if comp_alg in ["fedmax", "fedprox"]:
            local_epochs = 10
            total_rounds = 500
        else:
            local_epochs = 1
            total_rounds = 5000
        full_results = pd.DataFrame(columns=["Dataset", "Compression", "Aggregation", "Attack"] + [f"{a:.0%} Adv." for a in adv_percent])
        for dataset_name in ["mnist", "kddcup99"]:
            dataset = utils.load(dataset_name)
            for alg in ["foolsgold", "krum", "viceroy"]:
                for attack in ["labelflip", "onoff labelflip", "onoff freerider", "bad mouther"]:
                    cur = {"Dataset": dataset_name, "Compression": comp_alg, "Aggregation": alg, "Attack": attack}
                    for adv_p in adv_percent:
                        print(f"{dataset_name}, {comp_alg}-{alg}, {adv_p:.0%} {attack} adversaries")
                        if dataset_name == "kddcup99":
                            num_endpoints = 20
                            distribution = [[(i + 1 if i >= 11 else i) % dataset.classes, 11] for i in range(num_endpoints)]
                            attack_from, attack_to = 0, 11
                        else:
                            num_endpoints = 10
                            distribution = [[i % dataset.classes] for i in range(num_endpoints)]
                            attack_from, attack_to = 0, 1
                        victim = 0
                        num_adversaries = round(num_endpoints * adv_p)
                        num_clients = num_endpoints - num_adversaries

                        batch_sizes = [8 for _ in range(num_endpoints)]
                        data = dataset.fed_split(batch_sizes, distribution)
                        train_eval = dataset.get_iter("train", 10_000)
                        test_eval = dataset.get_iter("test")

                        selected_model = lambda: ymir.mp.models.LeNet_300_100(dataset.classes)
                        net = hk.without_apply_rng(hk.transform(lambda x: selected_model()(x)))
                        net_act = hk.without_apply_rng(hk.transform(lambda x: selected_model().act(x)))
                        opt = optax.sgd(0.01)
                        params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])
                        opt_state = opt.init(params)
                        if comp_alg == "fedmax":
                            loss = ymir.mp.losses.fedmax_loss(net, net_act, dataset.classes)
                            client_opt, client_opt_state = opt, opt_state
                        if comp_alg == "fedprox":
                            loss = ymir.mp.losses.cross_entropy_loss(net, dataset.classes)
                            client_opt = ymir.mp.optimizers.pgd(0.01, 1, local_epochs)
                            client_opt_state = client_opt.init(params)
                        else:
                            loss = ymir.mp.losses.cross_entropy_loss(net, dataset.classes)
                            client_opt, client_opt_state = opt, opt_state
    
                        network = ymir.mp.network.Network()
                        network.add_controller("main", server=True)
                        for i in range(num_clients):
                            network.add_host("main", ymir.scout.Collaborator(client_opt, client_opt_state, loss, data[i], local_epochs))

                        for i in range(num_adversaries):
                            c = ymir.scout.Collaborator(client_opt, client_opt_state, loss, data[i + num_clients], batch_sizes[i + num_clients])
                            if "labelflip" in attack:
                                ymir.scout.adversaries.labelflipper.convert(c, dataset, attack_from, attack_to)
                            elif "backdoor" in attack:
                                ymir.scout.adversaries.backdoor.convert(c, dataset, dataset_name, attack_from, attack_to)
                            elif "freerider" in attack:
                                ymir.scout.adversaries.freerider.convert(c, "delta", params)
                            if "onoff" in attack:
                                ymir.scout.adversaries.onoff.convert(c)
                            network.add_host("main", c)
                        controller = network.get_controller("main")
                        if "scaling" in attack:
                            controller.add_grad_transform(ymir.scout.adversaries.scaler.GradientTransform(network, params, alg, num_adversaries))
                        if "mouther" in attack:
                            controller.add_grad_transform(ymir.scout.adversaries.mouther.GradientTransform(num_adversaries, victim, num_adversaries))
                        if "onoff" not in attack:
                            toggler = None
                        else:
                            toggler = ymir.scout.adversaries.onoff.GradientTransform(
                                network, params, alg, controller.clients[-num_adversaries:],
                                max_alpha=1/num_endpoints if alg in ['fed_avg', 'std_dagmm'] else 1,
                                sharp=alg in ['fed_avg', 'std_dagmm', 'krum']
                            )
                            controller.add_grad_transform(toggler)
                        if comp_alg == "ae":
                            coder = ymir.mp.compression.ae.Coder(params, num_endpoints)
                            controller.add_grad_transform(ymir.mp.compression.ae.Encode(coder))
                            controller.add_grad_transform(ymir.mp.compression.ae.Decode(params, coder))
                        if comp_alg == "fedzip":
                            controller.add_grad_transform(ymir.mp.compression.fedzip.encode)
                            controller.add_grad_transform(ymir.mp.compression.fedzip.Decode(params))

                        model = ymir.Coordinate(alg, opt, opt_state, params, network)
                        meter = utils.Neurometer(
                            net,
                            {'train': train_eval, 'test': test_eval},
                            ['accuracy', 'asr'],
                            add_keys=['asr'] if "labelflip" not in attack else [],
                            attack_from=attack_from,
                            attack_to=attack_to
                        )

                        print("Done, beginning training.")

                        # Train/eval loop.
                        pbar = trange(total_rounds)
                        for _ in pbar:
                            attacking = toggler.attacking if toggler else True
                            if "labelflip" in attack:
                                results = meter.add_record(model.params)
                                pbar.set_postfix({'ACC': f"{results['test accuracy']:.3f}", 'ASR': f"{results['test asr']:.3f}", 'ATT': f"{attacking}"})
                            alpha, all_grads = model.step()
                            if "labelflip" not in attack:
                                if "freerider" in attack:
                                    if attacking:
                                        if alg == "krum":
                                            meter.add('asr', alpha[-num_adversaries:].mean())
                                        else:
                                            meter.add('asr', jnp.minimum(alpha[-num_adversaries:].mean() / (1 / (alpha > 0).sum()), 1))
                                    else:
                                        meter.add('asr', 0.0)
                                elif "mouther" in attack:
                                    if (alpha[-num_adversaries:] < 0.0001).all():
                                        asr = -1 if alpha[victim] < 0.0001 else -2
                                    else:
                                        theta = jax.flatten_util.ravel_pytree(ymir.garrison.sum_grads(all_grads))[0]
                                        vicdel = utils.euclid_dist(jax.flatten_util.ravel_pytree(all_grads[victim])[0], theta)
                                        if "good" in attack:
                                            numerator = min(utils.euclid_dist(jnp.array([jax.flatten_util.ravel_pytree(g)[0] for g in all_grads]), theta))
                                            asr = utils.unzero(numerator) / utils.unzero(vicdel)
                                        else:
                                            asr = utils.unzero(vicdel) / utils.unzero(max(utils.euclid_dist(jnp.array([jax.flatten_util.ravel_pytree(g)[0] for g in all_grads]), theta)))
                                    meter.add('asr', asr)
                                results = meter.add_record(model.params)
                                pbar.set_postfix({'ACC': f"{results['test accuracy']:.3f}", 'ASR': f"{results['asr']:.3f}", 'ATT': f"{attacking}"})
                        final_results = meter.get_results()
                        asrs = final_results['test asr'] if "labelflip" in attack else final_results['asr']
                        accs = meter.get_results()['test accuracy']
                        cur[f"{adv_p:.0%} Adv."] = f"ACC: {accs[-1]}, ASR: {asrs.mean()} ({asrs.std()})"
                        print(f"""Results are {cur[f"{adv_p:.0%} Adv."]}""")
                    full_results = full_results.append(cur, ignore_index=True)
        with pd.ExcelWriter("results.xlsx") as xls:
            full_results.to_excel(xls, sheet_name=comp_alg)


if __name__ == "__main__":
    app.run(main)