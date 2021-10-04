from dataclasses import dataclass

import jax
import jaxlib


"""
Basic federated averaging proposed in https://arxiv.org/abs/1602.05629

Call order: server init -> scale
"""


@dataclass
class Server:
    batch_sizes: jaxlib.xla_extension.DeviceArray


def scale(batch_sizes):
    return jax.vmap(lambda b: b / batch_sizes.sum())(batch_sizes)