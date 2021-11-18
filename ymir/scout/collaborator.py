from functools import partial
import jax

class Collaborator:
    def __init__(self, opt, opt_state, loss, data, epochs):
        self.opt_state = opt_state
        self.data = data
        self.batch_size = data.batch_size
        self.epochs = epochs
        self.opt = opt
        self.loss = loss
        self.update = partial(update, opt, loss)


@partial(jax.jit, static_argnums=(0, 1,))
def update(opt, loss, params, opt_state, X, y):
    grads = jax.grad(loss)(params, X, y)
    updates, opt_state = opt.update(grads, opt_state, params)
    return grads, opt_state, updates