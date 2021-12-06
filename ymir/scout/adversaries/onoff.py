from functools import partial

import numpy as np

from ymir import garrison
from ymir import mp
from ymir.scout import collaborator

class GradientTransform:
    """
    Network controller that toggles an attack on or off respective to the result of the aggregation algorithm
    """
    def __init__(self, params, opt, opt_state, network, alg, adversaries, max_alpha, sharp, beta=1.0, gamma=0.85, timer=False, rng=np.random.default_rng(), **kwargs):
        self.alg = alg
        self.attacking = False
        self.max_alpha = max_alpha
        self.sharp = sharp
        self.beta = beta
        self.gamma = gamma
        self.server = getattr(garrison, self.alg).Captain(params, opt, opt_state, network, rng, **kwargs)
        self.adversaries = adversaries
        self.num_adv = len(adversaries)
        self.timer_mode = timer
        if timer:
            self.timer = 0
        
    def should_toggle(self, alpha):
        if self.timer_mode:
            self.timer += 1
            if self.timer % 30 == 0:
                return True
            return False
        avg_syb_alpha = alpha[-self.num_adv:].mean()
        p = self.attacking and avg_syb_alpha < self.beta * self.max_alpha
        if self.sharp:
            q = not self.attacking and avg_syb_alpha > 0.4 * self.max_alpha
        else:
            q = not self.attacking and avg_syb_alpha > self.gamma * self.max_alpha
        return p or q

    def __call__(self, all_grads):
        """Update each connected client and return the generated gradients. Recursively call in connected controllers"""
        self.server.update(all_grads)
        alpha = self.server.scale(all_grads)
        if self.should_toggle(alpha):
            self.attacking = not self.attacking
            for a in self.adversaries:
                a.toggle()
        return all_grads


def convert(client):
    client.shadow_update = client.update
    client.update = partial(collaborator.update, client.opt, client.loss)
    client.toggle = toggle.__get__(client)


def toggle(self):
    self.update, shadow_update = self.shadow_update, self.update