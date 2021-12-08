"""
.. include:: README.md
"""

import numpy as np

from . import garrison
from . import mp
from . import regiment


class Coordinate:
    """Class for the high-level API for federated learning"""
    def __init__(self, alg, opt, opt_state, params, network, rng=np.random.default_rng(), **kwargs):
        """Construct the FL management object"""
        self.server = getattr(garrison, alg).Captain(params, opt, opt_state, network, rng, **kwargs)

    @property
    def params(self):
        return self.server.params

    @property
    def network(self):
        return self.server.network
    
    def step(self):
        """Perform a single round of federated learning"""
        return self.server.step()