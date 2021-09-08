import jax
import optax
import tensorflow_datasets as tfds

from . import aggregation
from . import compression

def load_dataset(split, batch_size, cache=True):
    """Loads the dataset as a generator of batches."""
    ds = tfds.load("mnist:3.*.*", split=split)
    if cache:
        ds = ds.cache().repeat()
    ds = ds.batch(batch_size)
    return iter(tfds.as_numpy(ds))

@jax.jit
def tree_mul(tree, scale):
    return jax.tree_map(lambda x: x * scale, tree)

@jax.jit
def tree_add(*trees):
    return jax.tree_multimap(lambda *xs: sum(xs), *trees)

def update(opt):
    @jax.jit
    def _apply(params, opt_state, grads):
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state
    return _apply

def apply_scale(alpha, all_grads):
    return [tree_mul(g, a) for g, a in zip(all_grads, alpha)]

def sum_grads(all_grads):
    return tree_add(*all_grads)
