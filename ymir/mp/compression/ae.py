import jax
import jax.numpy as jnp
import optax
import haiku as hk

import ymirlib

from .. import losses
from .. import network


class Controller(network.Controller):
    """
    Controller that performs AE on the gradients
    """
    def init(self, params):
        self.coder = ae_coder(params, len(self))

    def __call__(self, params):
        """Update each connected client and return the generated gradients. Recursively call in connected controllers"""
        all_grads = []
        for switch in self.switches:
            all_grads.extend(switch(params))
        for i, client in enumerate(self.clients):
            p = params
            sum_grads = None
            for _ in range(client.epochs):
                grads, client.opt_state, updates = self.update(p, client.opt_state, *next(client.data))
                p = optax.apply_updates(p, updates)
                sum_grads = grads if sum_grads is None else ymirlib.tree_add(sum_grads, grads)
                self.coder.add_data(jax.flatten_util.ravel_pytree(grads)[0], i)
            self.coder.update(i)
            sum_grads = self.coder.encode(jax.flatten_util.ravel_pytree(sum_grads)[0], i)
            all_grads.append(sum_grads)
        return all_grads


class Network(network.Network):
    """Network for handling FedZip"""
    def __call__(self, params):
        """Perform an update step across the network and return the respective gradients"""
        decoded_grads = self.controllers[self.server_name].coder.decode(self.controllers[self.server_name](params))
        unraveller = jax.flatten_util.ravel_pytree(params)[1]
        return [unraveller(d) for d in decoded_grads]


# Autoencoder compression: https://arxiv.org/abs/2108.05670

def _update(opt, loss):
    @jax.jit
    def _apply(params, opt_state, x):
        grads = jax.grad(loss)(params, x)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    return _apply


class ae_coder:
    def __init__(self, gm_params, num_clients):
        gm_params = jax.flatten_util.ravel_pytree(gm_params)[0]
        param_size = len(gm_params)
        ae = lambda: AE(param_size)
        self.f = hk.without_apply_rng(hk.transform(lambda x: ae()(x)))
        self.fe = hk.without_apply_rng(hk.transform(lambda x: ae().encode(x)))
        self.fd = hk.without_apply_rng(hk.transform(lambda x: ae().decode(x)))
        loss = losses.l2_loss(self.f)
        opt = optax.adam(1e-3)
        self.updater = _update(opt, loss)
        params = self.f.init(jax.random.PRNGKey(0), gm_params)
        self.params = [params for _ in range(num_clients)]
        self.opt_states = [opt.init(params) for _ in range(num_clients)]
        self.datas = [[] for _ in range(num_clients)]
        self.num_clients = num_clients

    def encode(self, grad, i):
        return self.fe.apply(self.params[i], jax.flatten_util.ravel_pytree(grad)[0])

    def decode(self, all_grads):
        return [self.fd.apply(self.params[i], grad) for i, grad in enumerate(all_grads)]

    def add_data(self, grad, i):
        self.datas[i].append(grad)
    
    def update(self, i):
        grads = jnp.array(self.datas[i])
        self.params[i], self.opt_states[i] = self.updater(self.params[i], self.opt_states[i], grads)
        self.datas[i] = []


class AE(hk.Module):
    def __init__(self, in_len, name=None):
        super().__init__(name=name)
        self.encoder = hk.Sequential([
            hk.Linear(60), jax.nn.relu,
            hk.Linear(30), jax.nn.relu,
            hk.Linear(10), jax.nn.relu,
            hk.Linear(1)
        ])
        self.decoder = hk.Sequential([
            hk.Linear(10), jax.nn.tanh,
            hk.Linear(30), jax.nn.tanh,
            hk.Linear(60), jax.nn.tanh,
            hk.Linear(in_len)
        ])
    
    def __call__(self, x):
        x = self.encoder(x)
        return self.decoder(x)
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)