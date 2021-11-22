import sklearn.metrics as skm

import numpy as np
import jax
import jax.numpy as jnp

"""
Measure performance during experiments
"""


def evaluator(net):
    """
    Get the tuple of of the form (y_true, y_preds) where the predictions are formed by the input net on the current params
    """
    @jax.jit
    def _apply(params, X, y):
        return (y, jnp.argmax(net.apply(params, X), axis=-1))
    return _apply

@jax.jit
def accuracy_score(y_true, y_pred):
    return jnp.mean(y_true == y_pred)


class Neurometer:
    """Measure aspects of the model"""
    def __init__(self, net, datasets):
        self.datasets = datasets
        self.results = {d: [] for d in datasets.keys()}
        self.evaluator = evaluator(net)
        self.classes = {d: ds.classes for d, ds in datasets.items()}
    
    def measure(self, params, accs: list = None):
        """Add a measurement of the chosen aspects with respect to the current params, return the latest results"""
        for ds_type, ds in self.datasets.items():
            self.results[ds_type].append(self.evaluator(params, *next(ds)))
        if accs is not None:
            return {a: accuracy_score(*self.results[a][-1]) for a in accs}

    def conclude(self):
        """Return overall results formatted into jax.numpy arrays"""
        for k, v in self.results.items():
            self.results[k] = [skm.confusion_matrix(y_true, y_pred, labels=range(self.classes[k])) for y_true, y_pred in v]
        return {k: np.array(v) for k, v in self.results.items()}