"""
Federated learning free rider attack proposed in `https://arxiv.org/abs/1911.12560 <https://arxiv.org/abs/1911.12560>`_
"""

import numpy as np

import ymir.path


def convert(client, attack_type, params, rng=np.random.default_rng()):
    """
    Convert an endpoint into a free rider adversary.

    Arguments:
    - client: the endpoint to convert
    - attack_type: the attack type to use, options are "random", "delta, and "advanced delta"
    - params: the parameters of the starting model
    - rng: the random number generator to use
    """
    client.attack_type = attack_type
    client.prev_params = params
    client.rng = rng
    client.update = update(client.opt).__get__(client)


def update(opt):
    """Free rider update function for endpoints."""

    def _apply(self, params, opt_state, X, y):
        if self.attack_type == "random":
            new_params = ymir.path.tree_uniform(params, low=-10e-3, high=10e-3, rng=self.rng)
        else:
            new_params = self.prev_params  # since at the higher level, the gradient is calculated as start_params - updated_params
            if "advanced" in self.attack_type:
                new_params = ymir.path.tree_add_normal(new_params, loc=0.0, scale=10e-4, rng=self.rng)
        self.prev_params = params
        return new_params, opt_state

    return _apply
