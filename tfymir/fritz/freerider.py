"""
Federated learning free rider attack proposed in `https://arxiv.org/abs/1911.12560 <https://arxiv.org/abs/1911.12560>`_
"""

import numpy as np

import tfymir.path


def convert(client, attack_type, params, rng=np.random.default_rng()):
    """
    Convert a client into a free rider adversary.

    Arguments:
    - client: the client to convert
    - attack_type: the attack type to use, options are "random", "delta, and "advanced delta"
    - params: the parameters of the starting model
    - rng: the random number generator to use
    """
    client.attack_type = attack_type
    client.prev_params = params
    client.rng = rng
    client.update = update.__get__(client)
    client.step = step.__get__(client)


def update(self, *_):
    """Stub for non-existent free-rider update function."""
    pass


def step(self, params, return_weights=False):
    """
    Perform a single local training loop.

    Arguments:
    - params: the parameters of the global model from the most recent round
    - return_weights: if True, return the weights of the clients else return the gradients from the local training
    """
    if self.attack_type == "random":
        grad = tfymir.path.tree.uniform(params, low=-10e-3, high=10e-3, rng=self.rng)
    else:
        grad = tfymir.path.tree.sub(params, self.prev_params)
        if "advanced" in self.attack_type:
            grad = tfymir.path.tree.add_normal(grad, loc=0.0, scale=10e-4, rng=self.rng)
    self.prev_params = params
    return tfymir.path.tree.add(params, grad) if return_weights else grad
