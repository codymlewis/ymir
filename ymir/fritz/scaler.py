"""
Scale the updates submitted from selected clients.
"""

import ymir.path


def convert(client, num_clients):
    """Scaled model replacement attack."""
    client.quantum_step = client.step
    client.step = lambda w, r: _scale(num_clients, *client.quantum_step(w, r))


def _scale(scale, loss, updates, batch_size):
    return loss, ymir.path.weights.scale(updates, scale), batch_size