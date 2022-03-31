"""
Example of federated averaging on the MNIST dataset
"""

import tensorflow as tf
import numpy as np
import tenjin
from tqdm import trange

import ymir

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
    dataset = ymir.mp.datasets.Dataset(*tenjin.load('mnist'))
    batch_sizes = [8 for _ in range(num_clients)]
    data = dataset.fed_split(batch_sizes, ymir.mp.distributions.lda, rng)
    train_eval = dataset.get_iter("train", 10_000, rng=rng)
    test_eval = dataset.get_iter("test", 10_000, rng=rng)

    # Setup the network
    network = ymir.mp.network.Network()
    for d in data:
        network.add_client(
            ymir.regiment.Scout(
                create_model(dataset.input_shape, dataset.classes), 
                d,
                tf.keras.optimizers.SGD(learning_rate=0.01),
                tf.keras.losses.SparseCategoricalCrossentropy(),
                1
            )
        )

    model = create_model(dataset.input_shape, dataset.classes)
    model.build()
    learner = ymir.garrison.fedavg.Captain(model.get_weights(), network, rng)
    print("Done, beginning training.")

    # Train/eval loop.
    for r in (pbar := trange(5000)):
        if r % 10 == 0:
            model.set_weights(learner.params)
            acc = ymir.mp.metrics.accuracy(model, test_eval)
            pbar.set_postfix({'ACC': f"{acc:.3f}"})
        loss = learner.step()
        pbar.set_postfix({'loss': f"{loss:.3f}"})
