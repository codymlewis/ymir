"""
Example of federated averaging on the MNIST dataset
"""

import tensorflow as tf
import haiku as hk
import hkzoo
import jax
import numpy as np
import optax
import tenjin
from tqdm import trange

import tfymir

def create_model(input_shape, output_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(output_shape, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    print("Setting up the system...")
    num_clients = 10
    rng = np.random.default_rng(0)

    # Setup the dataset
    dataset = tfymir.mp.datasets.Dataset(*tenjin.load('mnist'))
    batch_sizes = [8 for _ in range(num_clients)]
    data = dataset.fed_split(batch_sizes, tfymir.mp.distributions.lda, rng)
    train_eval = dataset.get_iter("train", 10_000, rng=rng)
    test_eval = dataset.get_iter("test", rng=rng)

    # Setup the network
    model = create_model(dataset.input_shape, dataset.classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    params = model.get_weights()
    network = tfymir.mp.network.Network()
    network.add_controller("main", server=True)
    for b, d in zip(batch_sizes, data):
        network.add_host("main", tfymir.regiment.Scout(model.copy(), *d, b, 1))

    server_opt = optax.sgd(1)
    server_opt_state = server_opt.init(params)
    model = tfymir.garrison.fedavg.Captain(params, server_opt, server_opt_state, network, rng)
    meter = tfymir.mp.metrics.Neurometer(model, {'train': train_eval, 'test': test_eval})

    print("Done, beginning training.")

    # Train/eval loop.
    for r in (pbar := trange(5000)):
        if r % 10 == 0:
            results = meter.measure(model.params, ['test'])
            pbar.set_postfix({'ACC': f"{results['test acc']:.3f}"})
        model.step()
