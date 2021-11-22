r"""
.. include:: README.md
"""

import numpy as np

from . import garrison
from . import mp
from . import scout


class Coordinate:
    """Class for the high-level API for federated learning"""
    def __init__(self, alg, opt, opt_state, params, network, rng=np.random.default_rng(), **kwargs):
        self.server = getattr(garrison.aggregators, alg).Server(params, network, **kwargs)
        self.params = params
        self.opt_state = opt_state
        self.server_update = garrison.update(opt)
        self.network = network
        self.rng = rng
    
    def step(self):
        """Perform a single round of federated learning"""
        # Client side updates
        all_grads = self.network(self.params, self.rng)

        # Server side aggregation scaling
        self.server.update(all_grads)
        alpha = self.server.scale(all_grads)
        all_grads = garrison.apply_scale(alpha, all_grads)

        # Server side update
        self.params, self.opt_state = self.server_update(self.params, self.opt_state, garrison.sum_grads(all_grads))
        return alpha, all_grads