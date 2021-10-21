import itertools
import sklearn.metrics as skm

import numpy as np
import jax
import jax.numpy as jnp

"""
Measure performance during experiments
"""


def accuracy(net, **_):
    """Find the accuracy of the models predictions on the data"""
    @jax.jit
    def _apply(params, X, y):
        return jnp.mean(jnp.argmax(net.apply(params, X), axis=-1) == y)
    return _apply

def asr(net, attack_from, attack_to, **_):
    """Find the success rate of a label flipping/backdoor attack that attempts the mapping attack_from -> attack_to"""
    @jax.jit
    def _apply(params, X, y):
        preds = jnp.argmax(net.apply(params, X), axis=-1)
        idx = y == attack_from
        return jnp.sum(jnp.where(idx, preds, -1) == attack_to) / jnp.sum(idx)
    return _apply


class Neurometer:
    """Measure aspects of the model"""
    def __init__(self, net, datasets, evals, **kwargs):
        self.datasets = datasets
        self.evaluators = {e: globals()[e](net, **kwargs) for e in evals}
        self.results = {f"{d} {e}": [] for d, e in itertools.product(datasets.keys(), evals)}

    def add_record(self, params):
        """Add a measurement of the chosen aspects with respect to the current params, return the latest results"""
        for ds_type, ds in self.datasets.items():
            for eval_type, eval in self.evaluators.items():
                self.results[f"{ds_type} {eval_type}"].append(eval(params, *next(ds)))
        return {k: v[-1] for k, v in self.results.items()}

    def get_results(self):
        """Return overall results formatted into jax.numpy arrays"""
        for k, v in self.results.items():
            self.results[k] = jnp.array(v)
        return self.results