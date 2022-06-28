"""
Federated learning free rider attack proposed in `https://arxiv.org/abs/1911.12560 <https://arxiv.org/abs/1911.12560>`_
"""

import numpy as np

from ymir.utils import functions


def convert(client, attack_type, rng=np.random.default_rng()):
    """
    Convert a client into a free rider adversary.
    Arguments:
    - client: the client to convert
    - attack_type: the attack type to use, options are "random", "delta, and "advanced delta"
    - rng: the random number generator to use
    """
    client.attack_type = attack_type
    client.prev_params = functions.ravel(client.params)
    client.rng = rng
    client.step = step.__get__(client)


def step(self, params, return_weights=False):
    """
    Perform a single local training loop.
    Arguments:
    - params: the parameters of the global model from the most recent round
    - return_weights: if True, return the weights of the clients else return the gradients from the local training
    """
    ravelled_params = functions.ravel(params)
    if self.attack_type == "random":
        grad = self.rng.uniform(size=self.params.shape, low=-1e-3, high=1e-3)
    else:
        grad = ravelled_params - self.prev_params
        if "advanced" in self.attack_type:
            grad += self.rng.normal(size=grad.shape, loc=0.0, scale=1e-4)
    self.prev_params = ravelled_params
    return ravelled_params + grad if return_weights else grad, 0.1, self.batch_size
