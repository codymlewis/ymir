"""
Set-up a network architecture for the FL process
"""

from ymir import scout


class Controller:
    def __init__(self, opt, loss):
        self.clients = []
        self.switches = []
        self.opt = opt
        self.loss = loss
        self.update = scout.update(opt, loss)

    def add_client(self, client):
        self.clients.append(client)

    def add_switch(self, switch):
        self.switches.append(switch)

    def __call__(self, params):
        all_grads = []
        for switch in self.switches:
            all_grads.extend(switch(params))
        for client in self.clients:
            grads, client.opt_state = self.update(params, client.opt_state, *next(client.data))
            all_grads.append(grads)
        return all_grads


class Network:
    def __init__(self, opt, loss):
        self.clients = []
        self.controllers = {}
        self.server_name = ""
        self.opt = opt
        self.loss = loss

    def __len__(self):
        return len(self.clients)

    def add_controller(self, name, is_server=False):
        self.controllers[name] = Controller(self.opt, self.loss)
        if is_server:
            self.server_name = name

    def add_host(self, controller_name, client):
        self.clients.append(client)
        self.controllers[controller_name].add_client(client)

    def connect_controllers(self, from_con, to_con):
        self.controllers[from_con].add_switch(self.controllers[to_con])

    def __call__(self, params):
        return self.controllers[self.server_name](params)

