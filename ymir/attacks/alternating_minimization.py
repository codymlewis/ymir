"""
Alternating minimization model poisoning, proposed in `https://arxiv.org/abs/1811.12470 <https://arxiv.org/abs/1811.12470>`_
"""

from ymir.client import client as ymir_client


def convert(client, poison_epochs, stealth_epochs, stealth_data):
    """
    Convert a client into an alternating minimization adversary.
    
    Arguments:
    - client: the client to convert
    - poison_epochs: the number of epochs to run the poisoned training for
    - stealth_epochs: the number of epochs to run the stealth training for
    - stealth_data: a generator that yields the stealth data
    """
    client.poison_step = client.step
    client.stealth_step = ymir_client.train_step(client.opt, client.loss)
    client.poison_epochs = poison_epochs
    client.stealth_epochs = stealth_epochs
    client.stealth_data = stealth_data
    client.step = step.__get__(client)


def step(self, params, opt_state, X, y):
    """Alternating minimization update function for clients."""
    for _ in range(self.poison_epochs):
        params, opt_state = self.poison_step(params, opt_state, X, y)
    for _ in range(self.stealth_epochs):
        params, opt_state = self.stealth_step(params, opt_state, *next(self.stealth_data))
    return params, opt_state
