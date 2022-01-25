"""
Alternating minimization model poisoning, proposed in `https://arxiv.org/abs/1811.12470 <https://arxiv.org/abs/1811.12470>`_
"""

from functools import partial

import ymir.path
import ymir.regiment.scout


def convert(client, poison_epochs, stealth_epochs, stealth_data):
    """
    Convert an endpoint into an alternating minimization adversary.
    
    Arguments:
    - client: the endpoint to convert
    - poison_epochs: the number of epochs to run the poisoned training for
    - stealth_epochs: the number of epochs to run the stealth training for
    - stealth_data: a generator that yields the stealth data
    """
    client.poison_update = client.update
    client.stealth_update = partial(ymir.regiment.scout.update, client.opt, client.loss)
    client.poison_epochs = poison_epochs
    client.stealth_epochs = stealth_epochs
    client.stealth_data = stealth_data
    client.update = update.__get__(client)


def update(self, params, opt_state, X, y):
    """Alternating minimization update function for endpoints."""
    for _ in range(self.poison_epochs):
        params, opt_state = self.poison_update(params, opt_state, X, y)
    for _ in range(self.stealth_epochs):
        params, opt_state = self.stealth_update(params, opt_state, *next(self.stealth_data))
    return params, opt_state