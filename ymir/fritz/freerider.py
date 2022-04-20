"""
Federated learning free rider attack proposed in `https://arxiv.org/abs/1911.12560 <https://arxiv.org/abs/1911.12560>`_
"""

import numpy as np

import ymir.path


def convert(client, attack_type, rng=np.random.default_rng()):
    """
    Convert a client into a free rider adversary.

    Arguments:
    - client: the client to convert
    - attack_type: the attack type to use, options are "random", "delta, and "advanced delta"
    - weights: the parameters of the starting model
    - rng: the random number generator to use
    """
    client.attack_type = attack_type
    client.prev_weights = client.model.get_weights()
    client.rng = rng
    client.step = step.__get__(client)


def step(self, weights, return_weights=False):
    """
    Perform a single local training loop.

    Arguments:
    - weights: the parameters of the global model from the most recent round
    - return_weights: if True, return the weights of the clients else return the gradients from the local training
    """
    if self.attack_type == "random":
        grad = ymir.path.weights.uniform(weights, low=-10e-3, high=10e-3, rng=self.rng)
    else:
        grad = ymir.path.weights.sub(weights, self.prev_weights)
        if "advanced" in self.attack_type:
            grad = ymir.path.weights.add_normal(grad, loc=0.0, scale=10e-4, rng=self.rng)
    self.prev_weights = weights
    return 0.1, ymir.path.weights.add(weights, grad) if return_weights else grad, self.batch_size
