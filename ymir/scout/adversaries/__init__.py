from dataclasses import dataclass
from typing import Mapping

from ymir import garrison
from ymir import mp

import numpy as np
import jax
import jax.numpy as jnp
import optax

from functools import partial
import itertools


@jax.jit
def tree_mul(tree, scale):
    return jax.tree_map(lambda x: x * scale, tree)


@jax.jit
def tree_rand(tree):
    return jax.tree_map(lambda x: np.random.uniform(low=-10e-3, high=10e-3, size=x.shape), tree)

@jax.jit
def tree_add_rand(tree):
    return jax.tree_map(lambda x: x + np.random.normal(loc=0.0, scale=10e-4, size=x.shape), tree)

@jax.jit
def tree_mul(tree, scale):
    """Multiply the elements of a pytree by the value of scale"""
    return jax.tree_map(lambda x: x * scale, tree)

@jax.jit
def tree_add(*trees):
    """Element-wise add any number of pytrees"""
    return jax.tree_multimap(lambda *xs: sum(xs), *trees)


def server_init(alg, params, clients):
    return {
        'fed_avg': lambda: garrison.aggregation.fed_avg.Server(jnp.array([c.batch_size for c in clients])),
        'foolsgold': lambda: garrison.aggregation.foolsgold.Server(len(clients), params, 1.0),
        'krum': lambda: None,
        'viceroy': lambda: garrison.aggregation.viceroy.Server(len(clients), params),
        'std_dagmm': lambda: garrison.aggregation.std_dagmm.Server(jnp.array([c.batch_size for c in clients]), params)
    }[alg]()


def update(alg, server, all_grads):
    if alg == 'foolsgold':
        server.histories = garrison.aggregation.foolsgold.update(server.histories, all_grads)
    elif alg == 'std_dagmm':
        server.params, server.opt_state = garrison.aggregation.std_dagmm.update(server, all_grads)
    elif alg == 'viceroy':
        server.histories, server.reps = garrison.aggregation.viceroy.update(server.histories, server.reps, all_grads)
    return server


def scale(alg, server, all_grads):
    if alg == 'fed_avg':
        alpha = garrison.aggregation.fed_avg.scale(server.batch_sizes)
    elif alg == 'foolsgold':
        alpha = garrison.aggregation.foolsgold.scale(server.histories, server.kappa)
    elif alg == 'krum':
        alpha = garrison.aggregation.krum.scale(all_grads, 3)
    elif alg == 'std_dagmm':
        alpha = garrison.aggregation.std_dagmm.scale(server.batch_sizes, all_grads, server)
    else:
        alpha = garrison.aggregation.viceroy.scale(server.histories, server.reps)
    return alpha


class ScalingController(mp.network.Controller):
    """
    Network controller that scales adversaries' gradients by the inverse of aggregation algorithm
    """
    def __init__(self, opt, loss, alg, num_adversaries):
        super().__init__(opt, loss)
        self.num_adv = num_adversaries
        self.alg = alg
        self.attacking = True

    def init(self, params):
        self.server = server_init(self.alg, params, self.clients)
        self.server_update = garrison.update(self.opt)
        
    def __call__(self, params):
        """Update each connected client and return the generated gradients. Recursively call in connected controllers"""
        all_grads = super().__call__(params)
        self.server = update(self.alg, self.server, all_grads)
        alpha = np.array(scale(self.alg, self.server, all_grads))
        idx = np.arange(len(alpha) - self.num_adv, len(alpha))[alpha[-self.num_adv:] > 0.0001]
        alpha[idx] = 1 / alpha[idx]
        for i in idx:
            all_grads[i] = tree_mul(all_grads[i], alpha[i])
        return all_grads

class OnOffController(mp.network.Controller):
    """
    Network controller that toggles an attack on or off respective to the result of the aggregation algorithm
    """
    def __init__(self, opt, loss, alg, num_adversaries, max_alpha, sharp, beta=1.0, gamma=0.85):
        super().__init__(opt, loss)
        self.num_adv = num_adversaries
        self.alg = alg
        self.attacking = False
        self.max_alpha = max_alpha
        self.sharp = sharp
        self.beta = beta
        self.gamma = gamma

    def init(self, params):
        self.server = server_init(self.alg, params, self.clients)
        self.server_update = garrison.update(self.opt)
        
    def should_toggle(self, alpha):
        avg_syb_alpha = alpha[-self.num_adv:].mean()
        p = self.attacking and avg_syb_alpha < self.beta * self.max_alpha
        if self.sharp:
            q = not self.attacking and avg_syb_alpha > 0.4 * self.max_alpha
        else:
            q = not self.attacking and avg_syb_alpha > self.gamma * self.max_alpha
        return p or q

    def __call__(self, params):
        """Update each connected client and return the generated gradients. Recursively call in connected controllers"""
        all_grads = super().__call__(params)
        self.server = update(self.alg, self.server, all_grads)
        alpha = scale(self.alg, self.server, all_grads)
        if self.should_toggle(alpha):
            self.attacking = not self.attacking
            for a in self.clients[-self.num_adv:]:
                a.toggle()
        return all_grads


class FRController(mp.network.Controller):
    """
    Network controller that that makes adversaries free ride
    """
    def __init__(self, opt, loss, num_adversaries, params, attack_type):
        super().__init__(opt, loss)
        self.num_adv = num_adversaries
        self.attacking = True
        self.prev_params = params
        self.attack_type = attack_type
        
    def __call__(self, params):
        """Update each connected client and return the generated gradients. Recursively call in connected controllers"""
        all_grads = super().__call__(params)
        if self.attack_type == "random":
            delta = tree_rand(params)
        else:
            delta = tree_add(params, tree_mul(self.prev_params, -1))
            if "advanced" in self.attack_type:
                delta = tree_add_rand(delta)
        all_grads[-self.num_adv:] = [delta for _ in range(self.num_adv)]
        self.prev_params = params
        return all_grads


class OnOffFRController(mp.network.Controller):
    """
    Network controller that that makes adversaries free ride respective to the results of the aggregation algorithm
    """
    def __init__(self, opt, loss, alg, num_adversaries, params, max_alpha, sharp, beta=1.0, gamma=0.85):
        super().__init__(opt, loss)
        self.num_adv = num_adversaries
        self.prev_params = params
        self.alg = alg
        self.attacking = False
        self.max_alpha = max_alpha
        self.sharp = sharp
        self.timer = 0
        self.beta = beta
        self.gamma = gamma

    def init(self):
        self.server = server_init(self.alg, self.prev_params, self.clients)
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

    def __call__(self, params):
        """Update each connected client and return the generated gradients. Recursively call in connected controllers"""
        all_grads = super().__call__(params)
        self.server = update(self.alg, self.server, all_grads)
        alpha = scale(self.alg, self.server, all_grads)
        if self.should_toggle(alpha):
            self.attacking = not self.attacking
        if self.attacking:
            delta = tree_add(params, tree_mul(self.prev_params, -1))
            all_grads[-self.num_adv:] = [delta for _ in range(self.num_adv)]
        self.prev_params = params
        return all_grads


class MoutherController(mp.network.Controller):
    """
    Network controller that scales adversaries' gradients by the inverse of aggregation algorithm
    """
    def __init__(self, opt, loss, num_adversaries, victim, attack_type):
        super().__init__(opt, loss)
        self.num_adv = num_adversaries
        self.attacking = True
        self.victim = victim
        self.attack_type = attack_type
        
    def __call__(self, params):
        """Update each connected client and return the generated gradients. Recursively call in connected controllers"""
        all_grads = super().__call__(params)
        grad = all_grads[self.victim]
        if "bad" in self.attack_type:
            grad = tree_mul(grad, -1)
        all_grads[-self.num_adv:] = [tree_add_rand(grad) for _ in range(self.num_adv)]
        return all_grads


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
        self.map=partial({"MNIST": mnist_backdoor_map, "CIFAR10": cifar10_backdoor_map, "KDDCup99": kddcup_backdoor_map}[type(dataset).__name__], attack_from, attack_to)
        self.data = dataset.get_iter(
            "train",
            batch_size,
            filter=lambda y: y == attack_from,
            map=self.map
        )
        self.batch_size = batch_size


def mnist_backdoor_map(attack_from, attack_to, X, y, no_label=False):
    idx = y == attack_from
    X[idx, 0:5, 0:5] = 1
    # X[idx][:, list(itertools.chain(*[list(range(x * 28, x * 28 + 5)) for x in range(5)]))] = 1
    if not no_label:
        y[idx] = attack_to
    return (X, y)

def cifar10_backdoor_map(attack_from, attack_to, X, y, no_label=False):
    idx = y == attack_from
    X[idx, 0:5, 0:5] = 1
    if not no_label:
        y[idx] = attack_to
    return (X, y)


def kddcup_backdoor_map(attack_from, attack_to, X, y, no_label=False):
    idx = y == attack_from
    X[idx, 0:5] = 1
    if not no_label:
        y[idx] = attack_to
    return (X, y)


def labelflip_map(attack_from, attack_to, X, y):
    idfrom = y == attack_from
    y[idfrom] = attack_to
    return (X, y)