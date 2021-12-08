import numpy as np

from ymir import garrison

import ymirlib


def convert(client, num_endpoints):
    """A simple naive scaled model replacement attack."""
    client.quantum_update = client.update
    client.update = lambda p, o, X, y: update(client.opt, 1/num_endpoints, p, o, client.quantum_update(p, o, X, y))


def update(opt, scale, params, opt_state, grads):
    grads = ymirlib.tree_mul(grads, scale)
    updates, opt_state = opt.update(grads, opt_state, params)
    return grads, opt_state, updates


class GradientTransform:
    """
    Network controller that scales adversaries' gradients by the inverse of aggregation algorithm
    """
    def __init__(self, params, opt, opt_state, network, alg, num_adversaries, rng = np.random.default_rng(), **kwargs):
        self.num_adv = num_adversaries
        self.alg = alg
        self.server = getattr(garrison, self.alg).Captain(params, opt, opt_state, network, rng, **kwargs)

    def __call__(self, all_grads):
        """Update each connected client and return the generated gradients. Recursively call in connected controllers"""
        self.server.update(all_grads)
        alpha = np.array(self.server.scale(all_grads))
        idx = np.arange(len(alpha) - self.num_adv, len(alpha))[alpha[-self.num_adv:] > 0.0001]
        alpha[idx] = 1 / alpha[idx]
        for i in idx:
            all_grads[i] = ymirlib.tree_mul(all_grads[i], alpha[i])
        return all_grads