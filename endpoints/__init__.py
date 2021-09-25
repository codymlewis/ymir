from dataclasses import dataclass
from typing import Mapping

import numpy as np
import jax
import optax

from . import adversaries
from . import compression


def update(opt, loss):
    @jax.jit
    def _apply(params, opt_state, batch):
        grads = jax.grad(loss)(params, batch)
        _, opt_state = opt.update(grads, opt_state)
        return grads, opt_state
    return _apply

@dataclass
class Client:
    opt_state: optax.OptState
    data: Mapping[str, np.ndarray]
    batch_size: int

    # def __init__(self, opt_state, data, batch_size):
    #     self.opt_state = opt_state
    #     self.data = data
    #     self.batch_size = batch_size