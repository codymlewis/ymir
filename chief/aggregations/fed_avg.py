from dataclasses import dataclass

import jax
import jaxlib


@dataclass
class Server:
    batch_sizes: jaxlib.xla_extension.DeviceArray


def scale(batch_sizes):
    return jax.vmap(lambda b: b / batch_sizes.sum())(batch_sizes)