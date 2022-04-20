"""
Alternating minimization model poisoning, proposed in `https://arxiv.org/abs/1811.12470 <https://arxiv.org/abs/1811.12470>`_
"""

import ymir.path
from ymir.regiment import scout


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
    client.stealth_step = scout.Scout.step.__get__(client)
    client.poison_epochs = poison_epochs
    client.stealth_epochs = stealth_epochs
    client.poison_data = client.data
    client.stealth_data = stealth_data
    client.step = step.__get__(client)


def step(self, weights, return_weights=False):
    """Alternating minimization update function for clients."""
    for _ in range(self.poison_epochs):
        self.data = self.poison_data
        self.poison_update(weights, return_weights)
    for _ in range(self.stealth_epochs):
        self.data = self.stealth_data
        loss, _, _ = self.stealth_update(self.model.get_weights(), return_weights)
    updates = self.model.get_weights() if return_weights else ymir.path.weights.sub(weights, self.model.get_weights())
    return loss, updates, self.batch_size
