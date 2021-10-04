import sys
import ymir.garrison
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import jax
import optax

from functools import partial
import itertools


""""
Endpoint-side adversaries for use. Requires a controller and the adversary endpoints
"""



class ScalingController:
    def __init__(self, num_adversaries, clients):
        self.num_adv = num_adversaries
        self.attacking = True

    def intercept(self, alpha, all_grads):
        all_grads[-self.num_adv:] = ymir.garrison.apply_scale(
            1/np.where(alpha[-self.num_adv:] == 0, sys.float_info.epsilon, alpha[-self.num_adv:]),
            all_grads[-self.num_adv:]
        )

class OnOffController:
    def __init__(self, num_adversaries, clients, max_alpha, sharp):
        self.num_adv = num_adversaries
        self.adv = clients[-num_adversaries:]
        self.attacking = False
        self.max_alpha = max_alpha
        self.sharp = sharp

    def should_toggle(self, alpha, beta, gamma):
        avg_syb_alpha = alpha[-self.num_adv:].mean()
        p = self.attacking and avg_syb_alpha < beta * self.max_alpha
        if self.sharp:
            q = not self.attacking and avg_syb_alpha > 0.4 * self.max_alpha
        else:
            q = not self.attacking and avg_syb_alpha > gamma * self.max_alpha
        return p or q

    # def intercept(self, alpha, all_grads, beta=1.0, gamma=0.85):
    def intercept(self, alpha, all_grads, beta=0.6, gamma=0.5):
        if self.should_toggle(alpha, beta, gamma):
            self.attacking = not self.attacking
            for a in self.adv:
                a.toggle()

class FRController:
    def __init__(self, num_adversaries, params, attack_type):
        self.num_adv = num_adversaries
        self.attacking = True
        self.prev_params = params
        self.attack_type = attack_type

    def intercept(self, alpha, all_grads, params):
        if self.attack_type == "random":
            delta = tree_rand(params)
        else:
            delta = ymir.garrison.tree_add(params, ymir.garrison.tree_mul(self.prev_params, -1))
            if "advanced" in self.attack_type:
                delta = tree_add_rand(delta)
        all_grads[-self.num_adv:] = [delta for _ in range(self.num_adv)]
        self.prev_params = params


class OnOffFRController:
    def __init__(self, num_adversaries, params, attack_type, clients, max_alpha, sharp):
        self.num_adv = num_adversaries
        self.adv = clients[-num_adversaries:]
        self.attacking = False
        self.max_alpha = max_alpha
        self.sharp = sharp
        self.prev_params = params
        self.attack_type = attack_type
        self.timer = 0

    def should_toggle(self, alpha, beta=1.0, gamma=0.85): # 0.7, 0.7
        avg_syb_alpha = alpha[-self.num_adv:].mean()
        p = self.attacking and avg_syb_alpha < beta * self.max_alpha
        if self.sharp:
            self.timer += 1
            return self.timer % 30
            # p = self.attacking and avg_syb_alpha < 0.7 * self.max_alpha
            # q = not self.attacking and avg_syb_alpha > 0.4 * self.max_alpha
        else:
            q = not self.attacking and avg_syb_alpha > gamma * self.max_alpha
        return p or q

    def intercept(self, alpha, all_grads, params):
        if self.attacking:
            if self.attack_type == "random":
                delta = tree_rand(params)
            else:
                delta = ymir.garrison.tree_add(params, ymir.garrison.tree_mul(self.prev_params, -1))
                if "advanced" in self.attack_type:
                    delta = tree_add_rand(delta)
            all_grads[-self.num_adv:] = [delta for _ in range(self.num_adv)]
        self.prev_params = params
        if self.should_toggle(alpha):
            self.attacking = not self.attacking

class MoutherController:
    def __init__(self, num_adversaries, victim, attack_type):
        self.num_adv = num_adversaries
        self.attacking = True
        self.victim = victim
        self.attack_type = attack_type

    def intercept(self, all_grads):
        grad = all_grads[self.victim]
        if "bad" in self.attack_type:
            grad = ymir.garrison.tree_mul(grad, -1)
        all_grads[-self.num_adv:] = [grad for _ in range(self.num_adv)]


@jax.jit
def tree_rand(tree):
    return jax.tree_map(lambda x: np.random.uniform(low=-10e-3, high=10e-3, size=x.shape), tree)

@jax.jit
def tree_add_rand(tree):
    return jax.tree_map(lambda x: x + np.random.normal(loc=0.0, scale=10e-3, size=x.shape), tree)

@dataclass
class OnOffLabelFlipper:
    opt_state: optax.OptState
    data: Mapping[str, np.ndarray]
    shadow_data: Mapping[str, np.ndarray]
    batch_size: int

    def __init__(self, opt_state, data, dataset, batch_size, attack_from, attack_to):
        self.opt_state = opt_state
        self.data = data
        self.shadow_data = dataset.get_iter(
            "train",
            batch_size,
            filter=lambda y: y == attack_from,
            map=partial(labelflip_map, attack_from, attack_to)
        )
        self.batch_size = batch_size

    def toggle(self):
        self.data, self.shadow_data = self.shadow_data, self.data

@dataclass
class OnOffFreeRider:
    opt_state: optax.OptState
    data: Mapping[str, np.ndarray]
    shadow_data: Mapping[str, np.ndarray]
    batch_size: int

    def __init__(self, opt_state, data, batch_size):
        self.opt_state = opt_state
        self.data = data
        self.batch_size = batch_size

    def toggle(self):
        self.data, self.shadow_data = self.shadow_data, self.data


class AdvController:
    def __init__(self, num_adversaries, clients):
        self.attacking = True

    def intercept(self, alpha, all_grads):
        pass

@dataclass
class LabelFlipper:
    opt_state: optax.OptState
    data: Mapping[str, np.ndarray]
    batch_size: int

    def __init__(self, opt_state, dataset, batch_size, attack_from, attack_to):
        self.opt_state = opt_state
        self.data = dataset.get_iter(
            "train",
            batch_size,
            filter=lambda y: y == attack_from,
            map=partial(labelflip_map, attack_from, attack_to)
        )
        self.batch_size = batch_size

@dataclass
class Backdoor:
    opt_state: optax.OptState
    data: Mapping[str, np.ndarray]
    batch_size: int

    def __init__(self, opt_state, dataset, batch_size, attack_from, attack_to):
        self.opt_state = opt_state
        self.data = dataset.get_iter(
            "train",
            batch_size,
            filter=lambda y: y == attack_from,
            map=partial(backdoor_map, attack_from, attack_to)
        )
        self.batch_size = batch_size


def backdoor_map(attack_from, attack_to, X, y):
    idx = y == attack_from
    X[idx][:, list(itertools.chain(*[list(range(x * 28, x * 28 + 5)) for x in range(5)]))] = 1
    # X[idx, 0:5, 0:5, 0] = 1
    y[idx] = attack_to
    return (X, y)


def labelflip_map(attack_from, attack_to, X, y):
    idx = y == attack_from
    y[idx] = attack_to
    return (X, y)