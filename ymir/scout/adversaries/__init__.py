from dataclasses import dataclass
from typing import Mapping

import ymirlib

from ymir import garrison
from ymir import mp

import numpy as np
import jax
import optax

from functools import partial


class ScalingController(mp.network.Controller):
    """
    Network controller that scales adversaries' gradients by the inverse of aggregation algorithm
    """
    def __init__(self, opt, loss, alg, C, num_adversaries):
        super().__init__(opt, loss, C)
        self.num_adv = num_adversaries
        self.alg = alg
        self.attacking = True

    def init(self, params):
        self.server = getattr(garrison.aggregators, self.alg).Server(params, self)
        self.server_update = garrison.update(self.opt)
        
    def __call__(self, params, rng=np.random.default_rng()):
        """Update each connected client and return the generated gradients. Recursively call in connected controllers"""
        all_grads = super().__call__(params, rng)
        self.server.update(all_grads)
        alpha = np.array(self.server.scale(all_grads))
        idx = np.arange(len(alpha) - self.num_adv, len(alpha))[alpha[-self.num_adv:] > 0.0001]
        alpha[idx] = 1 / alpha[idx]
        for i in idx:
            all_grads[i] = ymirlib.tree_mul(all_grads[i], alpha[i])
        return all_grads

class OnOffController(mp.network.Controller):
    """
    Network controller that toggles an attack on or off respective to the result of the aggregation algorithm
    """
    def __init__(self, opt, loss, C, alg, num_adversaries, max_alpha, sharp, beta=1.0, gamma=0.85):
        super().__init__(opt, loss, C)
        self.num_adv = num_adversaries
        self.alg = alg
        self.attacking = False
        self.max_alpha = max_alpha
        self.sharp = sharp
        self.beta = beta
        self.gamma = gamma

    def init(self, params):
        self.server = getattr(garrison.aggregators, self.alg).Server(params, self)
        self.server_update = garrison.update(self.opt)
        
    def should_toggle(self, alpha):
        avg_syb_alpha = alpha[-self.num_adv:].mean()
        p = self.attacking and avg_syb_alpha < self.beta * self.max_alpha
        if self.sharp:
            q = not self.attacking and avg_syb_alpha > 0.4 * self.max_alpha
        else:
            q = not self.attacking and avg_syb_alpha > self.gamma * self.max_alpha
        return p or q

    def __call__(self, params, rng=np.random.default_rng()):
        """Update each connected client and return the generated gradients. Recursively call in connected controllers"""
        all_grads = super().__call__(params, rng)
        self.server.update(all_grads)
        alpha = self.server.scale(all_grads)
        if self.should_toggle(alpha):
            self.attacking = not self.attacking
            for a in self.clients[-self.num_adv:]:
                a.toggle()
        return all_grads


class FRController(mp.network.Controller):
    """
    Network controller that that makes adversaries free ride
    """
    def __init__(self, opt, loss, C, num_adversaries, params, attack_type, rng=np.random.default_rng()):
        super().__init__(opt, loss, C)
        self.num_adv = num_adversaries
        self.attacking = True
        self.prev_params = params
        self.attack_type = attack_type
        self.rng = rng
        
    def __call__(self, params, rng=np.random.default_rng()):
        """Update each connected client and return the generated gradients. Recursively call in connected controllers"""
        all_grads = super().__call__(params, rng)
        if self.attack_type == "random":
            delta = ymirlib.tree_uniform(params, low=-10e-3, high=10e-3, rng=self.rng)
        else:
            delta = ymirlib.tree_add(params, ymirlib.tree_mul(self.prev_params, -1))
            if "advanced" in self.attack_type:
                delta = ymirlib.tree_add_normal(delta, loc=0.0, scale=10e-4, rng=self.rng)
        all_grads[-self.num_adv:] = [delta for _ in range(self.num_adv)]
        self.prev_params = params
        return all_grads


class OnOffFRController(mp.network.Controller):
    """
    Network controller that that makes adversaries free ride respective to the results of the aggregation algorithm
    """
    def __init__(self, opt, loss, C, alg, num_adversaries, params, max_alpha, sharp, beta=1.0, gamma=0.85):
        super().__init__(opt, loss, C)
        self.num_adv = num_adversaries
        self.prev_params = params
        self.alg = alg
        self.attacking = False
        self.max_alpha = max_alpha
        self.sharp = sharp
        self.timer = 0
        self.beta = beta
        self.gamma = gamma

    def init(self, params):
        self.server = getattr(garrison.aggregators, self.alg).Server(params, self)
        self.server_update = garrison.update(self.opt)
        
    def should_toggle(self, alpha): # 0.7, 0.7
        avg_syb_alpha = alpha[-self.num_adv:].mean()
        p = self.attacking and avg_syb_alpha < self.beta * self.max_alpha
        if self.sharp:
            self.timer += 1
            return self.timer % 30
        else:
            q = not self.attacking and avg_syb_alpha > self.gamma * self.max_alpha
        return p or q

    def __call__(self, params, rng=np.random.default_rng()):
        """Update each connected client and return the generated gradients. Recursively call in connected controllers"""
        all_grads = super().__call__(params, rng)
        self.server.update(all_grads)
        alpha = self.server.scale(all_grads)
        if self.should_toggle(alpha):
            self.attacking = not self.attacking
        if self.attacking:
            delta = ymirlib.tree_add(params, ymirlib.tree_mul(self.prev_params, -1))
            all_grads[-self.num_adv:] = [delta for _ in range(self.num_adv)]
        self.prev_params = params
        return all_grads


class MoutherController(mp.network.Controller):
    """
    Network controller that scales adversaries' gradients by the inverse of aggregation algorithm
    """
    def __init__(self, opt, loss, C, num_adversaries, victim, attack_type):
        super().__init__(opt, loss, C)
        self.num_adv = num_adversaries
        self.attacking = True
        self.victim = victim
        self.attack_type = attack_type
        
    def __call__(self, params, rng=np.random.default_rng()):
        """Update each connected client and return the generated gradients. Recursively call in connected controllers"""
        all_grads = super().__call__(params, rng)
        grad = all_grads[self.victim]
        if "bad" in self.attack_type:
            grad = ymirlib.tree_mul(grad, -1)
        all_grads[-self.num_adv:] = [ymirlib.tree_add_normal(grad, loc=0.0, scale=10e-4) for _ in range(self.num_adv)]
        return all_grads


@dataclass
class OnOffLabelFlipper:
    opt_state: optax.OptState
    data: Mapping[str, np.ndarray]
    shadow_data: Mapping[str, np.ndarray]
    batch_size: int
    epochs: int

    def __init__(self, opt_state, data, dataset, batch_size, epochs, attack_from, attack_to):
        self.opt_state = opt_state
        self.data = data
        self.shadow_data = dataset.get_iter(
            "train",
            batch_size,
            filter=lambda y: y == attack_from,
            map=partial(labelflip_map, attack_from, attack_to)
        )
        self.batch_size = batch_size
        self.epochs = epochs

    def toggle(self):
        self.data, self.shadow_data = self.shadow_data, self.data


@dataclass
class LabelFlipper:
    opt_state: optax.OptState
    data: Mapping[str, np.ndarray]
    batch_size: int
    epochs: int

    def __init__(self, opt_state, dataset, batch_size, epochs, attack_from, attack_to):
        self.opt_state = opt_state
        self.data = dataset.get_iter(
            "train",
            batch_size,
            filter=lambda y: y == attack_from,
            map=partial(labelflip_map, attack_from, attack_to)
        )
        self.batch_size = batch_size
        self.epochs = epochs

@dataclass
class Backdoor:
    opt_state: optax.OptState
    data: Mapping[str, np.ndarray]
    batch_size: int
    epochs: int

    def __init__(self, opt_state, dataset, batch_size, epochs, attack_from, attack_to):
        self.opt_state = opt_state
        self.map=partial(globals()[f"{type(dataset).__name__.lower()}_backdoor_map"], attack_from, attack_to)
        self.data = dataset.get_iter(
            "train",
            batch_size,
            filter=lambda y: y == attack_from,
            map=self.map
        )
        self.batch_size = batch_size
        self.epochs = epochs


def mnist_backdoor_map(attack_from, attack_to, X, y, no_label=False):
    idx = y == attack_from
    X[idx, 0:5, 0:5] = 1
    if not no_label:
        y[idx] = attack_to
    return (X, y)

def cifar10_backdoor_map(attack_from, attack_to, X, y, no_label=False):
    idx = y == attack_from
    X[idx, 0:5, 0:5] = 1
    if not no_label:
        y[idx] = attack_to
    return (X, y)


def kddcup99_backdoor_map(attack_from, attack_to, X, y, no_label=False):
    idx = y == attack_from
    X[idx, 0:5] = 1
    if not no_label:
        y[idx] = attack_to
    return (X, y)


def labelflip_map(attack_from, attack_to, X, y):
    idfrom = y == attack_from
    y[idfrom] = attack_to
    return (X, y)