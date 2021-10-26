import jax
import optax

import ymirlib

"""
Set-up a network architecture for the FL process
"""


def update(opt, loss):
    """Do some local training and return the gradient"""
    @jax.jit
    def _apply(params, opt_state, X, y):
        grads = jax.grad(loss)(params, X, y)
        updates, opt_state = opt.update(grads, opt_state)
        return grads, opt_state, updates
    return _apply


class Controller:
    """
    Holds a collection of clients and connects to other Controllers
    Handles the update step of each of the clients and passes the respective gradients
    up the chain.
    """
    def __init__(self, opt, loss):
        self.clients = []
        self.switches = []
        self.opt = opt
        self.loss = loss
        self.update = update(opt, loss)

    def __len__(self):
        return len(self.clients) + sum([len(s) for s in self.switches])

    def add_client(self, client):
        """Connect a client directly to this controller"""
        self.clients.append(client)

    def add_switch(self, switch):
        """Connect another controller to this controller"""
        self.switches.append(switch)

    def __call__(self, params):
        """Update each connected client and return the generated gradients. Recursively call in connected controllers"""
        all_grads = []
        for switch in self.switches:
            all_grads.extend(switch(params))
        for client in self.clients:
            p = params
            sum_grads = None
            for _ in range(client.epochs):
                grads, client.opt_state, updates = self.update(p, client.opt_state, *next(client.data))
                p = optax.apply_updates(p, updates)
                sum_grads = grads if sum_grads is None else ymirlib.tree_add(sum_grads, grads)
            all_grads.append(sum_grads)
        return all_grads


class Network:
    """Higher level class for tracking each controller and client"""
    def __init__(self, opt, loss):
        self.clients = []
        self.controllers = {}
        self.server_name = ""
        self.opt = opt
        self.loss = loss

    def __len__(self):
        """Get the number of clients in the network"""
        return len(self.clients)

    def add_controller(self, name, con_class=Controller, is_server=False):
        """Add a new controller with name into this network"""
        self.controllers[name] = con_class(self.opt, self.loss)
        if is_server:
            self.server_name = name
    
    def get_controller(self, name):
        return self.controllers[name]

    def add_host(self, controller_name, client):
        """Add a client to the specified controller in this network"""
        self.clients.append(client)
        self.controllers[controller_name].add_client(client)

    def connect_controllers(self, from_con, to_con):
        """Connect two controllers in this network"""
        self.controllers[from_con].add_switch(self.controllers[to_con])

    def __call__(self, params):
        """Perform an update step across the network and return the respective gradients"""
        return self.controllers[self.server_name](params)

