import jax
import jax.numpy as jnp
import numpy as np
from sklearn import cluster


def encode(grads, compress=True):
    usable_grads = jax.tree_leaves(jax.tree_map(lambda x: x.flatten(), grads))
    sparse_grads = [top_z(0.3, np.array(g)) for g in usable_grads]
    quantized_grads = [k_means(g) for g in sparse_grads]
    if compress:
        encoded_grads = []
        codings = []
        for g in quantized_grads:
            e = encoding(g)
            encoded_grads.append(e[0])
            codings.append(e[1])
        return encoded_grads, codings
    return jax.tree_multimap(lambda x, y: x.reshape(y.shape), jax.tree_unflatten(jax.tree_structure(grads), quantized_grads), grads)


def top_z(z, grads):
    z_index = np.ceil(z * grads.shape[0]).astype(np.int32)
    grads[np.argpartition(abs(grads), -z_index)[:-z_index]] = 0
    return grads

def k_means(grads):
    model = cluster.KMeans(init='random', n_clusters=3, max_iter=4, n_init=1, random_state=0)
    model.fit(np.unique(grads).reshape((-1, 1)))
    labels = model.predict(grads.reshape((-1, 1)))
    centroids = model.cluster_centers_
    for i, c in enumerate(centroids):
        grads[labels == i] = c[0]
    return grads

def encoding(grads):
    centroids = jnp.unique(grads).tolist()
    probs = []
    for c in centroids:
        probs.append(((grads == c).sum() / len(grads)).item())
    return huffman(grads, centroids, probs)

def huffman(grads, centroids, probs):
    groups = [(p, i) for i, p in enumerate(probs)]
    if len(centroids) > 1:
        while len(groups) > 1:
            groups.sort(key=lambda x: x[0])
            a, b = groups[0:2]
            del groups[0:2]
            groups.append((a[0] + b[0], [a[1], b[1]]))
        groups[0][1].sort(key=lambda x: isinstance(x, list))
        coding = {centroids[k]: v for (k, v) in  traverse_tree(groups[0][1])}
    else:
        coding = {centroids[0]: 0b0}
    result = jnp.zeros(grads.shape, dtype=jnp.int8)
    for c in centroids:
        result = jnp.where(grads == c, coding[c], result)
    return result, {v: k for k, v in coding.items()}


def traverse_tree(root, line=0b0):
    if isinstance(root, list):
        return traverse_tree(root[0], line << 1) + traverse_tree(root[1], (line << 1) + 0b1)
    return [(root, line)]
