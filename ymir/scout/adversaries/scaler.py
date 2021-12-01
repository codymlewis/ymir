import numpy as np

from ymir import garrison

import ymirlib


class GradientTransform:
    """
    Network controller that scales adversaries' gradients by the inverse of aggregation algorithm
    """
    def __init__(self, params, opt, opt_state, network, alg, num_adversaries, rng = np.random.default_rng(), **kwargs):
        self.num_adv = num_adversaries
        self.alg = alg
        self.server = getattr(garrison.aggregators, self.alg).Server(params, opt, opt_state, network, rng, **kwargs)

    def __call__(self, all_grads):
        """Update each connected client and return the generated gradients. Recursively call in connected controllers"""
        self.server.update(all_grads)
        alpha = np.array(self.server.scale(all_grads))
        idx = np.arange(len(alpha) - self.num_adv, len(alpha))[alpha[-self.num_adv:] > 0.0001]
        alpha[idx] = 1 / alpha[idx]
        for i in idx:
            all_grads[i] = ymirlib.tree_mul(all_grads[i], alpha[i])
        return all_grads