import jax.numpy as jnp


from . import garrison
from . import mp
from . import scout


"""
The generic high level API
"""


def server_init(alg, params, network):
    return {
        'fed_avg': lambda: garrison.aggregation.fed_avg.Server(jnp.array([c.batch_size for c in network.clients])),
        'foolsgold': lambda: garrison.aggregation.foolsgold.Server(len(network), params, 1.0),
        'krum': lambda: None,
        'viceroy': lambda: garrison.aggregation.viceroy.Server(len(network), params),
        'std_dagmm': lambda: garrison.aggregation.std_dagmm.Server(jnp.array([c.batch_size for c in network.clients]), params)
    }[alg]()


def update(alg, server, all_grads):
    if alg == 'foolsgold':
        server.histories = garrison.aggregation.foolsgold.update(server.histories, all_grads)
    elif alg == 'std_dagmm':
        server.params, server.opt_state = garrison.aggregation.std_dagmm.update(server, all_grads)
    elif alg == 'viceroy':
        server.histories, server.reps = garrison.aggregation.viceroy.update(server.histories, server.reps, all_grads)
    return server


def scale(alg, server, all_grads):
    if alg == 'fed_avg':
        alpha = garrison.aggregation.fed_avg.scale(server.batch_sizes)
    elif alg == 'foolsgold':
        alpha = garrison.aggregation.foolsgold.scale(server.histories, server.kappa)
    elif alg == 'krum':
        alpha = garrison.aggregation.krum.scale(all_grads, 3)
    elif alg == 'std_dagmm':
        alpha = garrison.aggregation.std_dagmm.scale(server.batch_sizes, all_grads, server)
    else:
        alpha = garrison.aggregation.viceroy.scale(server.histories, server.reps)
    return alpha



class Coordinate:
    """Class for the high-level API for federated learning"""
    def __init__(self, alg, opt, opt_state, params, network):
        self.alg = alg
        self.params = params
        self.opt_state = opt_state
        self.server = server_init(alg, params, network)
        self.server_update = garrison.update(opt)
        self.network = network
    
    def fit(self):
        """Perform a single round of federated learning"""
        # Client side updates
        all_grads = self.network(self.params)

        # Server side aggregation scaling
        self.server = update(self.alg, self.server, all_grads)
        alpha = scale(self.alg, self.server, all_grads)
        all_grads = garrison.apply_scale(alpha, all_grads)

        # Server side update
        self.params, self.opt_state = self.server_update(self.params, self.opt_state, garrison.sum_grads(all_grads))
        return alpha