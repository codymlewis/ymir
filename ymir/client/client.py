import jax
import optax

from ymir.utils import functions


class Client:

    def __init__(self, params, opt, loss, data, epochs=1):
        self._train_step = train_step(opt, loss)
        self.opt_state = opt.init(params)
        self.data = data
        self.epochs = epochs
        self.params = params

    def step(self, params, return_weights=False):
        self.params = params
        for e in range(self.epochs):
            X, y = next(self.data)
            self.params, self.opt_state, loss = self._train_step(self.params, self.opt_state, X, y)
        return functions.ravel(
            self.params
        ) if return_weights else functions.gradient(params, self.params), loss, self.data.batch_size


def train_step(opt, loss):

    @jax.jit
    def _apply(params, opt_state, X, y):
        loss_val, grads = jax.value_and_grad(loss)(params, X, y)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    return _apply
