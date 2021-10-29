import numpy as np
import jax
import jax.numpy as jnp
import optax
import haiku as hk

import ymirlib

from ymir import garrison

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


# Adversary compression controllers
class ScalingController(Controller):
    """
    Network controller that scales adversaries' gradients by the inverse of aggregation algorithm
    """
    def __init__(self, opt, loss, alg, num_adversaries):
        super().__init__(opt, loss)
        self.num_adv = num_adversaries
        self.alg = alg
        self.attacking = True

    def init(self, params):
        super().init(params)
        self.server = getattr(garrison.aggregation, self.alg).Server(params, self)
        self.server_update = garrison.update(self.opt)
        
    def __call__(self, params):
        """Update each connected client and return the generated gradients. Recursively call in connected controllers"""
        all_grads = super().__call__(params)
        # intercept
        decoded_grads = self.coder.decode(all_grads)
        unraveller = jax.flatten_util.ravel_pytree(params)[1]
        int_grads = [unraveller(d) for d in decoded_grads]
        self.server.update(int_grads)
        alpha = np.array(self.server.scale(int_grads))
        idx = np.arange(len(alpha) - self.num_adv, len(alpha))[alpha[-self.num_adv:] > 0.0001]
        alpha[idx] = 1 / alpha[idx]
        for i in idx:
            int_grads[i] = jax.flatten_util.ravel_pytree(ymirlib.tree_mul(int_grads[i], alpha[i]))[0]
            self.coder.add_data(jax.flatten_util.ravel_pytree(int_grads[i])[0], i)
            self.server.update(i)
            all_grads[i] = self.coder.encode(jax.flatten_util.ravel_pytree(int_grads[i])[0], i)
        return all_grads

class OnOffController(Controller):
    """
    Network controller that toggles an attack on or off respective to the result of the aggregation algorithm
    """
    def __init__(self, opt, loss, alg, num_adversaries, max_alpha, sharp, beta=1.0, gamma=0.85):
        super().__init__(opt, loss)
        self.num_adv = num_adversaries
        self.alg = alg
        self.attacking = False
        self.max_alpha = max_alpha
        self.sharp = sharp
        self.beta = beta
        self.gamma = gamma

    def init(self, params):
        super().init(params)
        self.server = getattr(garrison.aggregation, self.alg).Server(params, self)
        self.server_update = garrison.update(self.opt)
        
    def should_toggle(self, alpha):
        avg_syb_alpha = alpha[-self.num_adv:].mean()
        p = self.attacking and avg_syb_alpha < self.beta * self.max_alpha
        if self.sharp:
            q = not self.attacking and avg_syb_alpha > 0.4 * self.max_alpha
        else:
            q = not self.attacking and avg_syb_alpha > self.gamma * self.max_alpha
        return p or q

    def __call__(self, params):
        """Update each connected client and return the generated gradients. Recursively call in connected controllers"""
        all_grads = super().__call__(params)
        # intercept
        decoded_grads = self.coder.decode(all_grads)
        unraveller = jax.flatten_util.ravel_pytree(params)[1]
        int_grads = [unraveller(d) for d in decoded_grads]
        self.server.update(int_grads)
        alpha = self.server.scale(int_grads)
        if self.should_toggle(alpha):
            self.attacking = not self.attacking
            for a in self.clients[-self.num_adv:]:
                a.toggle()
        return all_grads


class FRController(network.Controller):
    """
    Network controller that that makes adversaries free ride
    """
    def __init__(self, opt, loss, num_adversaries, params, attack_type):
        super().__init__(opt, loss)
        self.num_adv = num_adversaries
        self.attacking = True
        self.prev_params = params
        self.attack_type = attack_type
        
    def __call__(self, params):
        """Update each connected client and return the generated gradients. Recursively call in connected controllers"""
        all_grads = super().__call__(params)
        # intercept
        decoded_grads = self.coder.decode(all_grads)
        unraveller = jax.flatten_util.ravel_pytree(params)[1]
        int_grads = [unraveller(d) for d in decoded_grads]
        if self.attack_type == "random":
            delta = ymirlib.tree_uniform(params, low=-10e-3, high=10e-3)
        else:
            delta = ymirlib.tree_add(params, ymirlib.tree_mul(self.prev_params, -1))
            if "advanced" in self.attack_type:
                delta = ymirlib.tree_add_normal(delta, loc=0.0, scale=10e-4)
        int_grads[-self.num_adv:] = [delta for _ in range(self.num_adv)]
        all_grads[-self.num_adv:] = [self.coder.encode(jax.flatten_util.ravel_pytree(delta)[0], i) for i, delta in enumerate(int_grads[-self.num_adv:])]
        self.prev_params = params
        return all_grads


class OnOffFRController(Controller):
    """
    Network controller that that makes adversaries free ride respective to the results of the aggregation algorithm
    """
    def __init__(self, opt, loss, alg, num_adversaries, params, max_alpha, sharp, beta=1.0, gamma=0.85):
        super().__init__(opt, loss)
        self.num_adv = num_adversaries
        self.prev_params = params
        self.alg = alg
        self.attacking = False
        self.max_alpha = max_alpha
        self.sharp = sharp
        self.timer = 0
        self.beta = beta
        self.gamma = gamma

    def init(self, params):
        super().init(params)
        self.server = getattr(garrison.aggregation, self.alg).Server(params, self)
        self.server_update = garrison.update(self.opt)
        
    def should_toggle(self, alpha):
        avg_syb_alpha = alpha[-self.num_adv:].mean()
        p = self.attacking and avg_syb_alpha < self.beta * self.max_alpha
        if self.sharp:
            self.timer += 1
            return self.timer % 30
        else:
            q = not self.attacking and avg_syb_alpha > self.gamma * self.max_alpha
        return p or q

    def __call__(self, params):
        """Update each connected client and return the generated gradients. Recursively call in connected controllers"""
        all_grads = super().__call__(params)
        # intercept
        decoded_grads = self.coder.decode(all_grads)
        unraveller = jax.flatten_util.ravel_pytree(params)[1]
        int_grads = [unraveller(d) for d in decoded_grads]
        self.server.update(int_grads)
        alpha = self.server.scale(int_grads)
        if self.should_toggle(alpha):
            self.attacking = not self.attacking
        if self.attacking:
            delta = ymirlib.tree_add(params, ymirlib.tree_mul(self.prev_params, -1))
            int_grads[-self.num_adv:] = [delta for _ in range(self.num_adv)]
        all_grads[-self.num_adv:] = [self.coder.encode(jax.flatten_util.ravel_pytree(delta)[0], i) for i, delta in enumerate(int_grads[-self.num_adv:])]
        self.prev_params = params
        return all_grads


class MoutherController(Controller):
    """
    Network controller that scales adversaries' gradients by the inverse of aggregation algorithm
    """
    def __init__(self, opt, loss, num_adversaries, victim, attack_type):
        super().__init__(opt, loss)
        self.num_adv = num_adversaries
        self.attacking = True
        self.victim = victim
        self.attack_type = attack_type
        
    def __call__(self, params):
        """Update each connected client and return the generated gradients. Recursively call in connected controllers"""
        all_grads = super().__call__(params)
        # intercept
        decoded_grads = self.coder.decode(all_grads)
        unraveller = jax.flatten_util.ravel_pytree(params)[1]
        int_grads = [unraveller(d) for d in decoded_grads]
        grad = int_grads[self.victim]
        if "bad" in self.attack_type:
            grad = ymirlib.tree_mul(grad, -1)
        int_grads[-self.num_adv:] = [ymirlib.tree_add_normal(grad, loc=0.0, scale=10e-4) for _ in range(self.num_adv)]
        all_grads[-self.num_adv:] = [self.coder.encode(jax.flatten_util.ravel_pytree(delta)[0], i) for i, delta in enumerate(int_grads[-self.num_adv:])]
        return all_grads