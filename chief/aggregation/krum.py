import numpy as np
import jax


def scale(grads, clip):
    n = len(grads)
    X = np.array([jax.flatten_util.ravel_pytree(g)[0] for g in grads])
    scores = np.zeros(n)
    distances = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None] - 2 * np.dot(X, X.T)
    for i in range(len(X)):
        scores[i] = np.sum(np.sort(distances[i])[1:((n - clip) - 1)])
    idx = np.argpartition(scores, n - clip)[:(n - clip)]
    alpha = np.zeros(n)
    alpha[idx] = 1
    return alpha