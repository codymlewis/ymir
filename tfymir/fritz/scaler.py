"""
Scale the updates submitted from selected clients.
"""

import jax
import numpy as np
import optax

import tfymir.path
from tfymir import garrison


def convert(client, num_clients):
    """A simple naive scaled model replacement attack."""
    client.quantum_update = client.update
    client.update = lambda p, o, X, y: _scale(num_clients, p, *client.quantum_update(p, o, X, y))


@jax.jit
def _scale(scale, global_params, client_params, opt_state):
    params = tfymir.path.tree.sub(client_params, global_params)
    params = tfymir.path.tree.scale(params, scale)
    return tfymir.path.tree.add(params, global_params), opt_state


class GradientTransform:
    """
    Gradient transform that scales updates based on the inverse of the result from the aggregation scale value.
    """

    def __init__(self, params, opt, opt_state, network, alg, num_adversaries, rng=np.random.default_rng(), **kwargs):
        """
        Construct the gradient transform.

        Arguments:
        - params: the parameters of the starting model
        - opt: the optimizer to use
        - opt_state: the optimizer state
        - network: the network of the FL environment
        - alg: the FL aggregation algorithm to use
        - num_adversaries: the number of adversaries
        - rng: the random number generator to use
        """
        self.num_adv = num_adversaries
        self.alg = alg
        self.server = getattr(garrison, self.alg).Captain(params, opt, opt_state, network, rng, **kwargs)

    def __call__(self, all_updates):
        """Get the scale value and scale the gradients."""
        self.server.update(all_updates)
        alpha = np.array(self.server.scale(all_updates))
        idx = np.arange(len(alpha) - self.num_adv, len(alpha))[alpha[-self.num_adv:] > 0.0001]
        alpha[idx] = 1 / alpha[idx]
        for i in idx:
            all_updates[i] = tfymir.path.tree.scale(all_updates[i], alpha[i])
        return all_updates
