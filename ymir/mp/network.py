"""
Set-up a network architecture for the FL process
"""

from ymir import scout


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
        self.update = scout.update(opt, loss)

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
            grads, client.opt_state = self.update(params, client.opt_state, *next(client.data))
            all_grads.append(grads)
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

    def add_controller(self, name, is_server=False):
        """Add a new controller with name into this network"""
        self.controllers[name] = Controller(self.opt, self.loss)
        if is_server:
            self.server_name = name

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

