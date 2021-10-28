import jax
import jax.numpy as jnp
import numpy as np
from sklearn import cluster
import optax

from ymir import garrison
import ymirlib

from .. import network

class Controller(network.Controller):
    """
    Controller that performs FedZip on the gradients
    """
    def init(self, params):
        pass

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
            sum_grads = encode(sum_grads)
            all_grads.append(sum_grads)
        return all_grads


class Network(network.Network):
    """Network for handling FedZip"""
    def __call__(self, params):
        """Perform an update step across the network and return the respective gradients"""
        return decode(params, self.controllers[self.server_name](params))


# FedZip: https://arxiv.org/abs/2102.01593
# Endpoint-side FedZip functionality


def encode(grads, compress=False):
    usable_grads = jax.tree_leaves(jax.tree_map(lambda x: x.flatten(), grads))
    sparse_grads = [_top_z(0.3, np.array(g)) for g in usable_grads]
    quantized_grads = [_k_means(g) for g in sparse_grads]
    if compress:
        encoded_grads = []
        codings = []
        for g in quantized_grads:
            e = _encoding(g)
            encoded_grads.append(e[0])
            codings.append(e[1])
        return encoded_grads, codings
    return jax.tree_multimap(lambda x, y: x.reshape(y.shape), jax.tree_unflatten(jax.tree_structure(grads), quantized_grads), grads)


def _top_z(z, grads):
    z_index = np.ceil(z * grads.shape[0]).astype(np.int32)
    grads[np.argpartition(abs(grads), -z_index)[:-z_index]] = 0
    return grads

def _k_means(grads):
    model = cluster.KMeans(init='random', n_clusters=3, max_iter=4, n_init=1, random_state=0)
    model.fit(np.unique(grads).reshape((-1, 1)))
    labels = model.predict(grads.reshape((-1, 1)))
    centroids = model.cluster_centers_
    for i, c in enumerate(centroids):
        grads[labels == i] = c[0]
    return grads

def _encoding(grads):
    centroids = jnp.unique(grads).tolist()
    probs = []
    for c in centroids:
        probs.append(((grads == c).sum() / len(grads)).item())
    return _huffman(grads, centroids, probs)

def _huffman(grads, centroids, probs):
    groups = [(p, i) for i, p in enumerate(probs)]
    if len(centroids) > 1:
        while len(groups) > 1:
            groups.sort(key=lambda x: x[0])
            a, b = groups[0:2]
            del groups[0:2]
            groups.append((a[0] + b[0], [a[1], b[1]]))
        groups[0][1].sort(key=lambda x: isinstance(x, list))
        coding = {centroids[k]: v for (k, v) in  _traverse_tree(groups[0][1])}
    else:
        coding = {centroids[0]: 0b0}
    result = jnp.zeros(grads.shape, dtype=jnp.int8)
    for c in centroids:
        result = jnp.where(grads == c, coding[c], result)
    return result, {v: k for k, v in coding.items()}


def _traverse_tree(root, line=0b0):
    if isinstance(root, list):
        return _traverse_tree(root[0], line << 1) + _traverse_tree(root[1], (line << 1) + 0b1)
    return [(root, line)]


# server-side FedZip functionality

def decode(params, all_grads, compress=False):
    if compress:
        return [_huffman_decode(params, g, e) for (g, e) in all_grads]
    return all_grads


@jax.jit
def _huffman_decode(params, grads, encodings):
    flat_params, tree_struct = jax.tree_flatten(params)
    final_grads = [jnp.zeros(p.shape, dtype=jnp.float32) for p in flat_params]
    for i, p in enumerate(flat_params):
        for k, v in encodings[i].items():
            final_grads[i] = jnp.where(grads[i].reshape(p.shape) == k, v, final_grads[i])
    return jax.tree_unflatten(tree_struct, final_grads)



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
        int_grads = decode(params, self.controllers[self.server_name](params))
        self.server.update(int_grads)
        alpha = np.array(self.server.scale(int_grads))
        idx = np.arange(len(alpha) - self.num_adv, len(alpha))[alpha[-self.num_adv:] > 0.0001]
        alpha[idx] = 1 / alpha[idx]
        for i in idx:
            int_grads[i] = jax.flatten_util.ravel_pytree(ymirlib.tree_mul(int_grads[i], alpha[i]))[0]
            all_grads[i] = encode(int_grads[i])
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
        int_grads = decode(params, all_grads)
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
        if self.attack_type == "random":
            delta = ymirlib.tree_uniform(params, low=-10e-3, high=10e-3)
        else:
            delta = ymirlib.tree_add(params, ymirlib.tree_mul(self.prev_params, -1))
            if "advanced" in self.attack_type:
                delta = ymirlib.tree_add_normal(delta, loc=0.0, scale=10e-4)
        all_grads[-self.num_adv:] = [encode(delta) for _ in range(self.num_adv)]
        self.prev_params = params
        return all_grads


class OnOffFRController(network.Controller):
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
        int_grads = decode(params, all_grads)
        self.server.update(int_grads)
        alpha = self.server.scale(int_grads)
        if self.should_toggle(alpha):
            self.attacking = not self.attacking
        if self.attacking:
            delta = ymirlib.tree_add(params, ymirlib.tree_mul(self.prev_params, -1))
            int_grads[-self.num_adv:] = [delta for _ in range(self.num_adv)]
        all_grads[-self.num_adv:] = [encode(g) for g in int_grads[-self.num_adv:]]
        self.prev_params = params
        return all_grads


class MoutherController(network.Controller):
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
        int_grads = decode(params, all_grads)
        grad = int_grads[self.victim]
        if "bad" in self.attack_type:
            grad = ymirlib.tree_mul(grad, -1)
        int_grads[-self.num_adv:] = [ymirlib.tree_add_normal(grad, loc=0.0, scale=10e-4) for _ in range(self.num_adv)]
        all_grads[-self.num_adv:] = [encode(g) for g in int_grads[-self.num_adv:]]
        return all_grads