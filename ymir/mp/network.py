"""
Defines the network architecture for the FL system.
"""

import numpy as np

import ymir.path


class Controller:
    """
    Holds a collection of clients and connects to other Controllers.  
    Handles the update step of each of the clients and passes the respective gradients
    up the chain.
    """

    def __init__(self, C):
        """
        Construct the Controller.

        Arguments:
        - C: percent of clients to randomly select for training at each round
        """
        self.clients = []
        self.switches = []
        self.C = C
        self.K = 0
        self.update_transform_chain = []
        self.metrics = []

    def __len__(self):
        return len(self.clients) + sum([len(s) for s in self.switches])

    def add_client(self, client):
        """Connect a client directly to this controller"""
        self.clients.append(client)
        self.K += 1

    def add_switch(self, switch):
        """Connect another controller (referred to as switch) to this controller"""
        self.switches.append(switch)

    def add_update_transform(self, update_transform):
        """Add a function that transforms the updates before passing them up the chain"""
        self.update_transform_chain.append(update_transform)

    def __call__(self, params, rng=np.random.default_rng(), return_weights=False):
        """
        Update each connected client and return the generated update. Recursively call in connected controllers
        
        Arguments:
        - params: the parameters of the global model from the most recent round
        - rng: the random number generator to use
        - return_weights: if True, return the weights of the clients else return the gradients from the local training
        """
        all_updates = []
        for switch in self.switches:
            all_updates.extend(switch(params, rng, return_weights))
        idx = rng.choice(self.K, size=int(self.C * self.K), replace=False) if self.C < 1 else range(self.K)
        for i in idx:
            all_updates.append(self.clients[i].step(params, return_weights))
        return ymir.path.functions.chain(self.update_transform_chain, all_updates)

    def add_metric(self, neurometer, data):
        """Add the specified metric measurement for each connection to this controller"""
        for switch in self.switches:
            switch.add_metric(neurometer, data)
        for c in self.clients:
            self.metrics.append(neurometer(c.net, data))

    def measure(self, accs=None, asrs=None):
        """Return the metrics for each connection to this controller"""
        results = [switch.measure(accs, asrs) for switch in self.switches]
        results += [m.measure(c.params, accs, asrs) for m, c in zip(self.metrics, self.clients)]
        return _merge_dicts(results)

    def conclude(self):
        """Conclude the metrics for each connection to this controller"""
        conclusions = [switch.conclude() for switch in self.switches]
        conclusions += [m.conclude() for m in self.metrics if m is not None]
        return _merge_dicts(conclusions)


def _merge_dicts(dicts):
    """Merge a list of dictionaries into a single dictionary, where values are a list of dict values"""
    if len(dicts):
        final_dicts = {k: [] for k in dicts[0].keys()}
    else:
        return {}
    for d in dicts:
        for k, v in d.items():
            final_dicts[k].append(v)
    return final_dicts


class Network:
    """Higher level class for tracking each controller and client"""

    def __init__(self, C=1.0):
        """Construct the Network.

        Arguments:
        - C: percent of clients to randomly select for training at each round
        """
        self.clients = []
        self.controllers = {}
        self.server_name = ""
        self.C = C

    def __len__(self):
        """Get the number of clients in the network"""
        return len(self.clients)

    def add_controller(self, name, server=False):
        """Add a new controller with name into this network"""
        self.controllers[name] = Controller(self.C)
        if server:
            self.server_name = name

    def get_controller(self, name):
        """Get the controller with the specified name"""
        return self.controllers[name]

    def add_host(self, controller_name, client):
        """Add a client to the specified controller in this network"""
        self.clients.append(client)
        self.controllers[controller_name].add_client(client)

    def connect_controllers(self, from_con, to_con):
        """Connect two controllers in this network"""
        self.controllers[from_con].add_switch(self.controllers[to_con])

    def __call__(self, params, rng=np.random.default_rng(), return_weights=False):
        """
        Perform an update step across the network and return the respective updates

        Arguments:
        - params: the parameters of the global model from the most recent round
        - rng: the random number generator to use
        - return_weights: if True, return the weights of the clients else return the gradients from the local training
        """
        return self.controllers[self.server_name](params, rng, return_weights)

    def add_metric(self, neurometer, data):
        self.controllers[self.server_name].add_metric(neurometer, data)

    def measure(self, accs=None, asrs=None):
        return self.controllers[self.server_name].measure(accs, asrs)

    def conclude(self):
        return self.controllers[self.server_name].conclude()
